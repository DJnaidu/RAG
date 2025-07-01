import os
import streamlit as st
from rag_engine import (
    insert_document,
    search_similar_documents,
    get_all_documents,
    update_document
)
from langchain_openai import ChatOpenAI

# Load OpenAI API key securely (for Streamlit Cloud, use secrets)
openai_key = st.secrets["OPENAI_API_KEY"]

# Streamlit UI setup
st.set_page_config(page_title="🧠 RAG Chatbot with Live Document Update", page_icon="📄")
st.title("📄 Upload or Edit TXT → Store in Supabase → Ask Questions")

# Upload .txt files
st.caption("📎 You can upload multiple .txt files. Keep each file below ~10MB for best results.")
uploaded_files = st.file_uploader("Upload .txt files", type="txt", accept_multiple_files=True)
if uploaded_files:
    for file in uploaded_files:
        content = file.read().decode("utf-8")
        insert_document(content)
    st.success("✅ Files uploaded and stored in Supabase!")

# Divider
st.markdown("---")
st.subheader("📝 Edit & Update Any Stored Document")

# Document Edit Section
try:
    all_docs = get_all_documents()
except Exception:
    st.error("⚠️ Failed to fetch documents from Supabase. Please check your database connection.")
    all_docs = []

if all_docs:
    doc_titles = {doc["content"][:50] + "..." : doc["id"] for doc in all_docs}
    selected_title = st.selectbox("📄 Select a document to edit", list(doc_titles.keys()))

    selected_id = doc_titles[selected_title]
    selected_doc = next((d for d in all_docs if d["id"] == selected_id), None)

    if selected_doc:
        updated_text = st.text_area("✏️ Edit Document", selected_doc["content"], height=300)
        if st.button("✅ Update Document"):
            update_document(selected_id, updated_text)
            st.success("✅ Document updated successfully!")
            st.rerun()  # Refresh the app to reflect changes

# Divider
st.markdown("---")
st.subheader("💬 Ask Questions")

query = st.text_input("🔍 Ask something from your uploaded documents:")
if query:
    with st.spinner("🔎 Searching Supabase..."):
        results = search_similar_documents(query, top_k=2)

        if results:
            context = "\n\n".join(res["content"] for res in results)

            prompt = f"""
You are a helpful assistant. Your job is to answer the user's question **using only the information provided in the context below**.

🚫 Do not use your training data.
🚫 Do not fact-check the context or override it.
✅ If the context says something factually wrong, treat it as true.

If the answer is not in the context, reply with:
"I don't know."

📄 Context:
{context}

❓Question:
{query}

💬 Answer:
"""

            llm = ChatOpenAI(temperature=0.3, openai_api_key=openai_key, model="gpt-3.5-turbo")
            answer = llm.invoke(prompt)

            st.markdown("### ✅ Answer")
            st.write(answer.content)
        else:
            st.warning("❌ No similar documents found.")
