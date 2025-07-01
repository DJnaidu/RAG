import os
from supabase import create_client, Client
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Insert document with embedding
def insert_document(content: str):
    embedding = model.encode(content).tolist()
    supabase.table("documents").insert({"content": content, "embedding": embedding}).execute()

# Get all documents
def get_all_documents():
    response = supabase.table("documents").select("id, content").execute()
    return response.data

# Update document and re-embed it
def update_document(doc_id: str, new_content: str):
    new_embedding = model.encode(new_content).tolist()
    supabase.table("documents").update({
        "content": new_content,
        "embedding": new_embedding
    }).eq("id", doc_id).execute()

# Search similar documents using the match_documents function
def search_similar_documents(query: str, top_k: int = 2):
    query_embedding = model.encode(query).tolist()
    response = supabase.rpc("match_documents", {
        "query_embedding": query_embedding,
        "match_count": top_k
    }).execute()
    return response.data
