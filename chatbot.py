import os
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv

# LangChain imports (correct provider packages)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Streamlit UI ---
st.header("📄 My First Chatbot")

with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF file and start asking questions", type="pdf")

# --- PDF Processing ---
if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Embeddings
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

    # Vector store
    vector_store = FAISS.from_texts(chunks, embeddings)

    # User question input
    user_question = st.text_input("💬 Type your question here")

    if user_question:
        # Find relevant docs
        matches = vector_store.similarity_search(user_question)

        # LLM
        llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            temperature=0,
            max_tokens=1000,
            model="gpt-3.5-turbo"
        )

        # QA chain
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=matches, question=user_question)

        # Output answer
        st.write(response)

else:
    st.write("👆 Upload a PDF to begin asking questions!")
