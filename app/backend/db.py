import os
from pymongo import MongoClient
from PyPDF2 import PdfReader
from datetime import datetime
from dotenv import load_dotenv
import numpy as np
from langchain_openai import AzureOpenAIEmbeddings

# Load environment variables from .env file
load_dotenv()

# MongoDB connection setup
mongo_uri = os.getenv("MONGO_CONNECTION_STRING")
db_name = os.getenv("MONGO_DB_NAME")
collection_name = os.getenv("MONGO_COLLECTION_NAME")
openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")

client = MongoClient(mongo_uri)
db = client[db_name]
collection = db[collection_name]

# Delete all existing documents
result = collection.delete_many({})
print(f"Deleted {result.deleted_count} documents from the collection.")

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Initialize the Azure OpenAI embeddings model
embeddings_model = AzureOpenAIEmbeddings(
    model="text-embedding-ada-002",  # Replace with your desired embedding model
    azure_deployment="text-embedding-ada-002",
    api_key=openai_api_key
)

# Directory where your PDF files are stored
pdf_directory = "../../data"

# Iterate through all files in the directory
for filename in os.listdir(pdf_directory):
    if filename.endswith(".pdf"):
        file_path = os.path.join(pdf_directory, filename)
        print(f"Processing file: {filename}")

        # Extract text from the PDF
        extracted_text = extract_text_from_pdf(file_path)

        # Generate embeddings for the extracted text
        embeddings = embeddings_model.embed_query(extracted_text)

        # Prepare the document for MongoDB insertion
        document = {
            "title": filename,
            "content": extracted_text,
            "embedding": embeddings,  # Store embeddings as a list of floats
            "metadata": {
                "upload_date": datetime.utcnow(),
                "source_file": filename
            }
        }

        # Insert the document into Cosmos DB
        collection.insert_one(document)
        print(f"Inserted document for {filename} into Cosmos DB")

print("All documents processed and inserted.")
