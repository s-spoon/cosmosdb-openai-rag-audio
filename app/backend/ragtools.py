from langchain.schema import Document
import re
import os
from dotenv import load_dotenv
from rtmt import Tool, ToolResult, ToolResultDirection
from pymongo import MongoClient
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores.azure_cosmos_db import (
    AzureCosmosDBVectorSearch,
    CosmosDBSimilarityType,
    CosmosDBVectorSearchType
)
import pdfplumber  # Library to extract text from PDF

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

# Function to extract text from PDF files in a directory
def extract_text_from_pdfs(pdf_dir):
    documents = []
    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            filepath = os.path.join(pdf_dir, filename)
            with pdfplumber.open(filepath) as pdf:
                full_text = ""
                for page in pdf.pages:
                    full_text += page.extract_text()
                # Ensure that we are creating Document objects
                documents.append(Document(page_content=full_text, metadata={"title": filename}))
    return documents

# Initialize MongoDB client
def init_mongo_client(mongo_connection_string):
    return MongoClient(mongo_connection_string)

# Vector search using CosmosDB Vector Store
def vector_search(query, vector_store):
    # Perform similarity search on the query
    docs = vector_store.similarity_search(query)
    return docs

def _search_tool(mongo_client, vector_store, args):
    query = args['query']
    print(f"Searching for '{query}' in the knowledge base.")

    # Perform vector search using CosmosDB vector store
    results = vector_search(query, vector_store)
    
    # Format results to be sent as a system message to the LLM
    result_str = ""
    for i, doc in enumerate(results):
        truncated_content = doc.page_content[:2000] if len(doc.page_content) > 2000 else doc.page_content
        result_str += f"[doc_{i}]: {doc.metadata['title']}\nContent: {truncated_content}\n-----\n"
    if not result_str or result_str.isspace():
        result_str = "1"
    
    return ToolResult(result_str, ToolResultDirection.TO_SERVER)

def _report_grounding_tool(mongo_client, vector_store, args):
    sources = args["sources"]
    valid_sources = [s for s in sources if re.match(r'^[a-zA-Z0-9_=\-]+$', s)]
    list_of_sources = " OR ".join(valid_sources)
    print(f"Grounding source: {list_of_sources}")

    # Fetch documents from the vector store using the search functionality
    search_results = []

    for source in valid_sources:
        # Perform a vector search using the title as the query
        docs = vector_store.similarity_search(source)
        for doc in docs:
            search_results.append({
                "chunk_id": doc.metadata.get("title", "Unknown Title"),
                "content": doc.page_content,
                "metadata": doc.metadata
            })

    # Format the results
    result_str = ""
    for result in search_results:
        result_str += f"[{result['chunk_id']}]: {result['content'][:200]}...\n-----\n"
    
    if not result_str or result_str.isspace():
        result_str = "1"

    return ToolResult(result_str.strip(), ToolResultDirection.TO_SERVER)

# Attach RAG tools to the RTMiddleTier
def attach_rag_tools(rtmt, mongo_connection_string, database_name, collection_name, pdf_dir):
    # Initialize MongoDB client
    mongo_client = init_mongo_client(mongo_connection_string)

    # Initialize embedding model (Azure OpenAI)
    openai_embeddings = AzureOpenAIEmbeddings(
        model=os.getenv("AZURE_OPENAI_EMBEDDINGS_MODEL_NAME"),
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY")
    )

    # Extract text from PDFs and create embeddings
    documents = extract_text_from_pdfs(pdf_dir)

    # Create vector store using AzureCosmosDBVectorSearch
    collection = mongo_client[database_name][collection_name]
    index_name = "ContosoIndex"
    vector_store = AzureCosmosDBVectorSearch.from_documents(
        documents,
        openai_embeddings,
        collection=collection,
        index_name=index_name,
    )

    # Create HNSW index on the collection
    num_lists = 100
    dimensions = 1536  # Size of the embeddings
    similarity_algorithm = CosmosDBSimilarityType.COS
    kind = CosmosDBVectorSearchType.VECTOR_HNSW
    m = 16
    ef_construction = 64

    vector_store.create_index(
        num_lists, dimensions, similarity_algorithm, kind, m, ef_construction
    )

    # Attach search and grounding tools
    rtmt.tools["search"] = Tool(
        schema=_search_tool_schema,
        target=lambda args: _search_tool(mongo_client, vector_store, args)
    )
    rtmt.tools["report_grounding"] = Tool(
        schema=_grounding_tool_schema,
        target=lambda args: _report_grounding_tool(mongo_client, vector_store, args)
    )
