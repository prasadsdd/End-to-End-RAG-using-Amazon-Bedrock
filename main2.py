import os
import boto3
import streamlit as st
from langchain.llms.bedrock import Bedrock
from langchain.embeddings import BedrockEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from streamlit_lottie import st_lottie
import requests
import tempfile

st.set_page_config(page_title="End-to-End RAG using Bedrock", page_icon="📚", layout="wide")

load_dotenv()

aws_access_key_id = os.getenv("aws_access_key_id")
aws_secret_access_key = os.getenv("aws_secret_access_key")
region_name = os.getenv("region_name")

# Setup Bedrock client
bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name=region_name,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
)

bedrock_embedding = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

# Helper functions to load lottie animations
def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_animation_url = "https://assets8.lottiefiles.com/packages/lf20_fcfjwiyb.json"

# Custom Styles
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        color: white;
        background-color: #4CAF50;
        border-radius: 12px;
    }
    .stTextInput>div>div>input {
        background-color: #f5f5f5;
        color: #333;
    }
    </style>
    """,
    unsafe_allow_html=True
)

prompt_template = """
Human: Use the following pieces of context to provide a 
concise answer to the question at the end but use at least summarize with 
250 words with detailed explanations. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>
Question: {question}
Assistant:"""

# Functions for processing
def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name
    
    loader = PyPDFLoader(temp_file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
    docs = text_splitter.split_documents(documents)
    
    os.remove(temp_file_path)
    
    return docs

def get_vector_store(docs, pdf_name):
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embedding)
    vectorstore_faiss.save_local(f"faiss_local_{pdf_name}")

def get_llm():
    llm = Bedrock(model_id="meta.llama3-70b-instruct-v1:0", client=bedrock)
    return llm

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

def get_llm_response(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    response = qa({"query": query})
    return response['result']

# Main Streamlit app
def main():
    st.header("📚 End-to-End RAG using Bedrock")

    # Load and display animation
    lottie_animation = load_lottie_url(lottie_animation_url)
    if lottie_animation:
        st_lottie(lottie_animation, height=150)

    user_question = st.text_input("🔍 Ask a question from the PDF file")

    # Layout using columns
    col1, col2 = st.columns(2)

    with col1:
        st.sidebar.title("Upload and Manage PDF")
        uploaded_file = st.sidebar.file_uploader("📂 Upload a PDF", type="pdf")

        if uploaded_file is not None:
            pdf_name = uploaded_file.name.replace(".pdf", "")
            if st.sidebar.button("Store Vector"):
                with st.spinner("Processing..."):
                    docs = process_pdf(uploaded_file)
                    get_vector_store(docs, pdf_name)
                    st.sidebar.success("PDF processed and stored!")

    with col2:
        if st.button("Send"):
            if user_question:
                if uploaded_file is not None:
                    pdf_name = uploaded_file.name.replace(".pdf", "")
                    faiss_index = FAISS.load_local(f"faiss_local_{pdf_name}", bedrock_embedding, allow_dangerous_deserialization=True)
                    llm = get_llm()
                    answer = get_llm_response(llm, faiss_index, user_question)
                    st.write("### 🤖 Assistant Response")
                    st.write(answer)
                else:
                    st.warning("Please upload a PDF first.")
            else:
                st.warning("Please enter a question.")

if __name__ == "__main__":
    main()
