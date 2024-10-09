import re
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_openai import AzureOpenAIEmbeddings
from rtmt import Tool, ToolResult, ToolResultDirection
from sklearn.metrics.pairwise import cosine_similarity
from rtmt import RTMiddleTier
import numpy as np

load_dotenv()

_search_tool_schema = {
    "type": "function",
    "name": "search",
    "description": "Search the knowledge base. The knowledge base is in English, translate to and from English if " + \
                   "needed. Results are formatted as a source name first in square brackets, followed by the text " + \
                   "content, and a line with '-----' at the end of each result.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query"
            }
        },
        "required": ["query"],
        "additionalProperties": False
    }
}

_grounding_tool_schema = {
    "type": "function",
    "name": "report_grounding",
    "description": "Report use of a source from the knowledge base as part of an answer (effectively, cite the source). Sources " + \
                   "appear in square brackets before each knowledge base passage. Always use this tool to cite sources when responding " + \
                   "with information from the knowledge base.",
    "parameters": {
        "type": "object",
        "properties": {
            "sources": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "List of source names from last statement actually used, do not include the ones not used to formulate a response"
            }
        },
        "required": ["sources"],
        "additionalProperties": False
    }
}

# MongoDB setup and tool attachment function
def init_mongo_client(mongo_connection_string):
    return MongoClient(mongo_connection_string)

# Function to perform vector search using cosine similarity in MongoDB
async def vector_search(query, mongo_client, database_name, collection_name, embeddings_model):
    collection = mongo_client[database_name][collection_name]

    # Generate embedding for the query
    query_embedding = np.array(embeddings_model.embed_query(query))

    # Fetch all documents
    docs = collection.find({})

    results = []
    for doc in docs:
        doc_embedding = np.array(doc['embedding'])
        if doc_embedding.size == 0:
            continue  # Skip if no valid embedding is found
        # Calculate cosine similarity between query and document embeddings
        similarity = cosine_similarity([query_embedding], [doc_embedding])[0][0]
        results.append((doc['title'], similarity))  # Changed 'text' to 'title'

    # Sort by similarity score in descending order
    results.sort(key=lambda x: x[1], reverse=True)

    # Return top 5 most similar documents
    return results[:5]

async def _search_tool(mongo_client, database_name, collection_name, embeddings_model, args: any) -> ToolResult:
    print(f"Searching for '{args['query']}' in the knowledge base.")
    
    # Perform the vector search in MongoDB
    results = await vector_search(args['query'], mongo_client, database_name, collection_name, embeddings_model)
    
    result_str = ""
    async for i, (title, similarity) in enumerate(results):
        result_str += f"[doc_{i}]: {title} (Similarity: {similarity:.4f})\n-----\n"
    
    return ToolResult(result_str, ToolResultDirection.TO_SERVER)

KEY_PATTERN = re.compile(r'^[a-zA-Z0-9_=\-]+$')

async def _report_grounding_tool(mongo_client, database_name, collection_name, args: any) -> ToolResult:
    sources = [s for s in args["sources"] if KEY_PATTERN.match(s)]
    list_of_sources = " OR ".join(sources)
    print(f"Grounding source: {list_of_sources}")

    # Fetch documents from MongoDB based on the sources
    collection = mongo_client[database_name][collection_name]  # Ensure you have access to the correct DB and collection
    search_results = []

    # Look up each source in the MongoDB collection
    for source in sources:
        doc = await collection.find_one({"title": source})  # Adjust the field if needed (e.g., based on your document structure)
        if doc:
            search_results.append({
                "chunk_id": doc.get("title"),  # Assuming "title" is the identifier; replace if necessary
                "content": doc.get("content"),  # Fetch other fields as needed
                "metadata": doc.get("metadata")  # If you want to include metadata
            })

    # Format the results for the response
    result_str = ""
    async for result in search_results:
        result_str += f"[{result['chunk_id']}]: {result['content']}\n-----\n"

    return ToolResult(result_str.strip(), ToolResultDirection.TO_SERVER)


def attach_rag_tools(rtmt: RTMiddleTier, mongo_connection_string: str, database_name: str, collection_name: str) -> None:
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
