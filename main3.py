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
import tempfile

load_dotenv()

aws_access_key_id = os.getenv("aws_access_key_id")
aws_secret_access_key = os.getenv("aws_secret_access_key")
region_name = os.getenv("region_name")

prompt_template = """
Human: Use the following pieces of context to provide a 
concise answer to the question at the end but use at least summarize with 
250 words with detailed explanations. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context

Question: {question}

Assistant:"""

bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name=region_name,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
)

bedrock_embedding = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)


def process_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file.read())
        temp_file_path = temp_file.name

    loader = PyPDFLoader(temp_file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=500)
    docs = text_splitter.split_documents(documents)

    os.remove(temp_file_path)  # Clean up the temporary file
    return docs


def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(
        docs,
        bedrock_embedding
    )
    vectorstore_faiss.save_local("faiss_local")


def get_llm():
    llm = Bedrock(model_id="meta.llama3-70b-instruct-v1:0", client=bedrock)
    return llm


PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


def get_llm_response(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    response = qa({"query": query})
    return response['result']


def main():
    st.set_page_config("RAG")
    st.header("End-to-End RAG using Bedrock")

    with st.sidebar:
        st.title("Upload & Process PDF")

        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

        if uploaded_file is not None:
            st.success("PDF uploaded successfully")

        if st.button("Store Vector"):
            with st.spinner("Processing.."):
                if uploaded_file is not None:
                    docs = process_pdf(uploaded_file)
                    get_vector_store(docs)
                    st.success("Vectors stored successfully")
                else:
                    st.error("Please upload a PDF file first.")

    col1, col2 = st.columns(2)

    with col1:
        user_question = st.text_input("Ask a question from the PDF file")
        if st.button("Send"):
            with st.spinner("Processing.."):
                faiss_index = FAISS.load_local("faiss_local", bedrock_embedding, allow_dangerous_deserialization=True)
                llm = get_llm()
                response = get_llm_response(llm, faiss_index, user_question)
    
    with col2:
        if 'response' in locals():
            st.write(response)


if __name__ == "__main__":
    main()