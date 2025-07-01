import os
import streamlit as st
from dotenv import load_dotenv
from rag_engine import (
    insert_document,
    search_similar_documents,
    get_all_documents,
    update_document
)
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# Load env vars
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# UI setup
st.set_page_config(page_title="📄 RAG Chatbot", page_icon="🤖")
st.title("📄 Upload or Edit TXT → Store in Supabase → Ask Questions")

# Upload documents
uploaded_files = st.file_uploader("Upload .txt files", type="txt", accept_multiple_files=True)
if uploaded_files:
    for file in uploaded_files:
        content = file.read().decode("utf-8")
        insert_document(content)
    st.success("✅ Files uploaded and stored in Supabase!")

st.markdown("---")
st.subheader("📝 Edit & Update Any Stored Document")

# Edit section
all_docs = get_all_documents()
if all_docs:
    doc_map = {doc["content"][:60] + "..." : doc["id"] for doc in all_docs}
    selected_title = st.selectbox("📄 Select a document to edit", list(doc_map.keys()))
    selected_id = doc_map[selected_title]
    selected_doc = next(d for d in all_docs if d["id"] == selected_id)

    updated_text = st.text_area("✏️ Edit Document", selected_doc["content"], height=300)
    if st.button("✅ Update Document"):
        update_document(selected_id, updated_text)
        st.success("✅ Document updated successfully!")
        st.rerun()

st.markdown("---")
st.subheader("💬 Ask Questions")

query = st.text_input("🔍 Ask something from your uploaded documents:")
if query:
    with st.spinner("🔎 Searching Supabase..."):
        results = search_similar_documents(query, top_k=2)

        if results:
            context = "\n\n".join(res["content"] for res in results)

            # Optional: show retrieved context for debugging
            # st.subheader("🧾 Retrieved Context")
            # st.code(context)

            template = """You are a helpful assistant. Answer the question using **only** the context below.
If the answer is not in the context, reply with: "I don't know."

Context:
{context}

Question:
{question}

Answer:"""

            prompt = PromptTemplate.from_template(template)
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3, openai_api_key=openai_key)
            chain = prompt | llm
            answer = chain.invoke({"context": context, "question": query})

            st.markdown("### ✅ Answer")
            st.write(answer.content)
        else:
            st.warning("❌ No similar documents found.")
