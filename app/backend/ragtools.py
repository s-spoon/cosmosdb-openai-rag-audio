import re
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_openai import AzureOpenAIEmbeddings
from rtmt import Tool, ToolResult, ToolResultDirection
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load environment variables
load_dotenv()

# Search tool schema
_search_tool_schema = {
    "type": "function",
    "name": "search",
    "description": "Search the knowledge base using a query and return relevant results.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query"
            }
        },
        "required": ["query"]
    }
}

# Grounding tool schema
_grounding_tool_schema = {
    "type": "function",
    "name": "report_grounding",
    "description": "Report the use of a source from the knowledge base by citing it.",
    "parameters": {
        "type": "object",
        "properties": {
            "sources": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "List of source names that were used."
            }
        },
        "required": ["sources"]
    }
}

# Initialize MongoDB client
def init_mongo_client(mongo_connection_string):
    return MongoClient(mongo_connection_string)

# Vector search using cosine similarity on document content
def vector_search(query, mongo_client, database_name, collection_name, embeddings_model):
    collection = mongo_client[database_name][collection_name]

    # Generate embedding for the query
    query_embedding = np.array(embeddings_model.embed_query(query))

    # Fetch all documents from the collection
    docs = collection.find({})

    results = []
    for doc in docs:
        doc_embedding = np.array(doc['embedding'])  # Using content embedding instead of title
        if doc_embedding.size == 0:
            continue  # Skip if no valid embedding is found
        
        # Calculate cosine similarity between query and document embeddings
        similarity = cosine_similarity([query_embedding], [doc_embedding])[0][0]
        results.append((doc['title'], similarity, doc['content']))  # Include 'content' for detailed results

    # Sort by similarity score in descending order
    results.sort(key=lambda x: x[1], reverse=True)

    # Return the top 5 most similar documents
    return results[:5]

def _search_tool(mongo_client, database_name, collection_name, embeddings_model, args):
    query = args['query']
    print(f"Searching for '{query}' in the knowledge base.")

    # Perform vector search on content embeddings
    results = vector_search(query, mongo_client, database_name, collection_name, embeddings_model)
    
    # Format results to be sent as a system message to the LLM
    result_str = ""
    for i, (title, similarity, content) in enumerate(results):
        truncated_content = content[:2000] if len(content) > 2000 else content
        result_str += f"[doc_{i}]: {title} (Similarity: {similarity:.4f})\nContent: {truncated_content}\n-----\n"
    
    return ToolResult(result_str, ToolResultDirection.TO_SERVER)

def _report_grounding_tool(mongo_client, database_name, collection_name, args):
    sources = args["sources"]
    valid_sources = [s for s in sources if re.match(r'^[a-zA-Z0-9_=\-]+$', s)]
    list_of_sources = " OR ".join(valid_sources)
    print(f"Grounding source: {list_of_sources}")

    # Fetch documents from MongoDB based on the sources
    collection = mongo_client[database_name][collection_name]
    search_results = []

    for source in valid_sources:
        doc = collection.find_one({"title": source})
        if doc:
            search_results.append({
                "chunk_id": doc.get("title"),
                "content": doc.get("content"),
                "metadata": doc.get("metadata")
            })

    # Format the results
    result_str = ""
    for result in search_results:
        result_str += f"[{result['chunk_id']}]: {result['content'][:200]}...\n-----\n"

    return ToolResult(result_str.strip(), ToolResultDirection.TO_SERVER)

# Attach RAG tools to the RTMiddleTier
def attach_rag_tools(rtmt, mongo_connection_string, database_name, collection_name):
    # Initialize MongoDB client
    mongo_client = init_mongo_client(mongo_connection_string)

    # Initialize embedding model (Azure OpenAI)
    openai_embeddings = AzureOpenAIEmbeddings(
        model="text-embedding-ada-002",
        azure_deployment="text-embedding-ada-002",
        api_key=os.getenv("AZURE_OPENAI_API_KEY")
    )

    # Attach search and grounding tools
    rtmt.tools["search"] = Tool(
        schema=_search_tool_schema,
        target=lambda args: _search_tool(mongo_client, database_name, collection_name, openai_embeddings, args)
    )
    rtmt.tools["report_grounding"] = Tool(
        schema=_grounding_tool_schema,
        target=lambda args: _report_grounding_tool(mongo_client, database_name, collection_name, args)
    )
