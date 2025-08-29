import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv

# Load environment variables from .env file (if present)
load_dotenv()

# Get OpenAI API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
#OPENAI_API_KEY = "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"

if not OPENAI_API_KEY:
    st.error("⚠️ OpenAI API key not found! Please set OPENAI_API_KEY in environment or .env file.")
    st.stop()

#upload pdf file
st.header("MY First Chatbot")

with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF file and start asking questions", type="pdf")

#Extract the text
if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text+=page.extract_text() or ""
        #st.write(text)

    #Break it into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len    
    )
    chunks = text_splitter.split_text(text)
    #st.write(chunks)

    #Generating Embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    #creating vector store - FAISS → embeddings (OpenAI), initizling FAISS, store chunks & embeddings
    vector_store = FAISS.from_texts(chunks, embeddings)

    #User question
    user_question = st.text_input("Type your Question here")

    #Similarity search
    if user_question:
        match = vector_store.similarity_search(user_question)
        #st.write(match)

        #Define llm
        llm = ChatOpenAI(
            openai_api_key = OPENAI_API_KEY,
            temperature = 0,
            max_tokens = 1000,
            model_name = "gpt-3.5-turbo" 
        )

        #output results
        #chain →  take the question, get revelant dosument, pass it to LLM, generate the output
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents = match, question = user_question)
        st.write(response)
    
else:
    st.write("👆 Upload a PDF and ask a question above!")