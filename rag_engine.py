import os
from dotenv import load_dotenv
from supabase import create_client, Client
from langchain_openai import OpenAIEmbeddings

# Load environment variables
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Create Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Set up embedding model
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# ─────────────────────────────────────────────────────────────────────────────
# Insert new document into Supabase (with embedding)
def insert_document(content: str):
    embedding = embedding_model.embed_query(content)
    supabase.table("documents").insert({
        "content": content,
        "embedding": embedding
    }).execute()

# ─────────────────────────────────────────────────────────────────────────────
# Search similar documents using remote match_documents() function
def search_similar_documents(query: str, top_k: int = 2):
    query_embedding = embedding_model.embed_query(query)

    response = supabase.rpc("match_documents", {
        "query_embedding": query_embedding,
        "match_count": top_k
    }).execute()

    return response.data if hasattr(response, "data") else []

# ─────────────────────────────────────────────────────────────────────────────
# Get all stored documents (id + content)
def get_all_documents():
    response = supabase.table("documents").select("id, content").execute()
    return response.data if hasattr(response, "data") else []

# ─────────────────────────────────────────────────────────────────────────────
# Update document (including regenerated embedding)
def update_document(doc_id: str, new_content: str):
    new_embedding = embedding_model.embed_query(new_content)

    supabase.table("documents").update({
        "content": new_content,
        "embedding": new_embedding
    }).eq("id", doc_id).execute()
