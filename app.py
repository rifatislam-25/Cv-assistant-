import os
import streamlit as st
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq

# --- Secure API Key ---
os.environ["GROQ_API_KEY"] = "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"  # Replace with your actual key

# --- Load Embedding Model ---
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")

# --- PDF to Text Function ---
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# --- Streamlit App ---
def main():
    st.set_page_config(page_title="CV Assistant AI", page_icon="🧠")
    st.title("🧠 CV Assistant Chatbot")
    st.caption("Upload your CV PDF and ask anything about it!")

    uploaded_file = st.file_uploader("Upload your CV (PDF)", type=["pdf"])

    if uploaded_file is not None:
        with st.spinner("Extracting text from PDF..."):
            cv_text = extract_text_from_pdf(uploaded_file)

            # --- Chunking ---
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=300,
                chunk_overlap=50,
                separators=["\n\n", "\n", ". ", ", "]
            )
            cv_chunks = text_splitter.split_text(cv_text)

            # --- Embedding & Retrieval ---
            vector_store = FAISS.from_texts(cv_chunks, embedding_model)
            retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

            # --- Groq LLM ---
            llm = ChatGroq(
                api_key=os.environ["GROQ_API_KEY"],
                model="llama3-8b-8192",
                temperature=0.7
            )

            # --- Memory ---
            memory = ConversationBufferMemory(memory_key="chat_history", output_key="answer", return_messages=True)

            # --- QA Chain ---
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=memory,
                return_source_documents=True
            )

            st.success("✅ CV loaded! Start chatting below.")

            if "messages" not in st.session_state:
                st.session_state.messages = []

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if prompt := st.chat_input("Ask something about your CV..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        result = qa_chain({"question": prompt, "chat_history": qa_chain.memory.chat_memory})
                        response = result['answer']
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()


