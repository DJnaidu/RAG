import os
from dotenv import load_dotenv
from supabase import create_client, Client
from langchain.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Initialize Supabase and Embedding model
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Insert a document into Supabase with embedding
def insert_document(content: str):
    embedding = embedding_model.embed_query(content)
    embedding_str = ",".join(map(str, embedding))  # Convert to string for Supabase VECTOR
    supabase.table("documents").insert({
        "content": content,
        "embedding": embedding_str
    }).execute()

# Fetch all documents
def get_all_documents():
    response = supabase.table("documents").select("id, content").execute()
    return response.data if hasattr(response, "data") else []

# Update document content and its embedding
def update_document(doc_id: str, new_content: str):
    new_embedding = embedding_model.embed_query(new_content)
    embedding_str = ",".join(map(str, new_embedding))  # Ensure proper format
    supabase.table("documents").update({
        "content": new_content,
        "embedding": embedding_str
    }).eq("id", doc_id).execute()

# Search similar documents using Supabase RPC
def search_similar_documents(query: str, top_k: int = 2):
    query_embedding = embedding_model.embed_query(query)
    embedding_str = ",".join(map(str, query_embedding))

    response = supabase.rpc("match_documents", {
        "query_embedding": embedding_str,
        "match_count": top_k
    }).execute()

    return response.data if hasattr(response, "data") else []
