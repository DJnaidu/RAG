import os
from dotenv import load_dotenv
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
from typing import List

# Load environment variables
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Initialize Supabase client and embedding model
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Insert a new document
def insert_document(content: str):
    embedding = embedding_model.encode(content).tolist()
    response = supabase.table("documents").insert({
        "content": content,
        "embedding": embedding
    }).execute()
    return response

# Update document and its embedding
def update_document(doc_id, new_content):
    embedding = embedding_model.encode(new_content).tolist()
    print("🔄 Updated embedding:", embedding[:5])  # Debug
    response = supabase.table("documents").update({
        "content": new_content,
        "embedding": embedding
    }).eq("id", doc_id).execute()
    return response

# Get all documents (for edit UI)
def get_all_documents():
    response = supabase.table("documents").select("id, content").execute()
    return response.data or []

# Use pgvector-powered SQL function for matching
def search_similar_documents(query: str, top_k: int = 2) -> List[dict]:
    query_embedding = embedding_model.encode(query).tolist()
    print("🔍 Query embedding:", query_embedding[:5])  # Debug
    response = supabase.rpc(
        "match_documents",
        {
            "query_embedding": query_embedding,
            "match_count": top_k
        }
    ).execute()

    if response.data:
        return [
            {"score": row["similarity"], "content": row["content"]}
            for row in response.data
        ]
    return []
