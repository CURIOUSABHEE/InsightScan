import os
import streamlit as st
import pickle
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.document_loaders import UnstructuredURLLoader
from dotenv import load_dotenv
#
# load_dotenv()
#
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Check if the API key is loaded correctly
# if GEMINI_API_KEY is None:
#     raise ValueError("GEMINI_API_KEY is not set in the .env file or environment variables.")
#

st.title("News Analyser")
st.sidebar.title("Articles Links")

main_placeholder = st.empty()  # Define the placeholder

urls = []
for i in range(1):
    url = st.sidebar.text_input(f" URL {i + 1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URL")
file_path = "vector_index.pkl"
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key= "AIzaSyArX3F0UsS_j-mRZORJr9NLCwYedDXJXj0"
)

if process_url_clicked:
    # load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text('Data loading...')
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "."],
        chunk_size=1000
    )
    main_placeholder.text('Text Splitting...')
    chunks = text_splitter.split_documents(data)

    # create embedding
    embedding = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

    # Create FAISS index
    vectorstore_index = FAISS.from_documents(chunks, embedding)
    main_placeholder.text('Embedding started...')
    time.sleep(2)

    #Save the index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_index, f)

query = main_placeholder.text_input("Question : ")

if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever = vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
            st.header("Answer: ")
            st.write(result["answer"])

            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources")
                sources_list = sources.split("\n")
                for source in sources_list:
                    st.write(source)
