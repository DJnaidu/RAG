import os
from dotenv import load_dotenv
from supabase import create_client, Client
from langchain.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize embedding model (MiniLM)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Store a document and its embedding
def insert_document(content: str):
    embedding = embedding_model.embed_query(content)
    supabase.table("documents").insert({
        "content": content,
        "embedding": embedding  # Must be a list[float]
    }).execute()

# Get all documents for UI listing
def get_all_documents():
    response = supabase.table("documents").select("id, content").execute()
    return response.data

# Update a document's content and re-generate embedding
def update_document(doc_id: str, new_content: str):
    new_embedding = embedding_model.embed_query(new_content)
    supabase.table("documents").update({
        "content": new_content,
        "embedding": new_embedding
    }).eq("id", doc_id).execute()

# Search similar documents using Supabase RPC
def search_similar_documents(query: str, top_k: int = 2):
    query_embedding = embedding_model.embed_query(query)
    response = supabase.rpc(
        "match_documents",
        {
            "query_embedding": query_embedding,  # must be a list[float]
            "match_count": top_k
        }
    ).execute()
    return response.data
