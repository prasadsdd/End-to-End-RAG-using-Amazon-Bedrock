
---

# 🚀 End-to-End Retrieval-Augmented Generation (RAG) with Amazon Bedrock

Build your own RAG application from scratch using **Amazon Bedrock**! This guide will walk you through setting up the environment, running the application, and containerizing it with Docker.

---


## 🔄 RAG Operation Flow

This flowchart illustrates the **end-to-end process of RAG** in the application, showing the steps from document ingestion by the admin to user query handling. The process includes chunking, vectorizing, and storing document embeddings, as well as querying for relevant information and generating responses.

![Image_Alt](https://github.com/prasadsdd/End-to-End-RAG-using-Amazon-Bedrock/blob/e3084aa84721e599ccc7276d59be0774771e7fda/Screenshot%20(11).png)

---
## 💻 User Interface of the RAG Application

The user interface enables easy document upload and question-answering functionality. Users can upload a PDF file, ask questions related to its content, and receive contextual answers based on the embedded document data.

![Image_Alt]()


---
## 📖 What is RAG?

**Retrieval-Augmented Generation (RAG)** is a powerful method combining **retrieval** and **generation** for dynamic, fact-based AI responses. It fetches relevant information from a database (or knowledge base) and uses it to enhance text generation, making your application more **accurate** and **context-aware**.

---

## 🌟 Features

- **End-to-End Setup**: From environment creation to Docker deployment
- **Dynamic Query Responses**: Uses RAG to generate responses with real-time data
- **Amazon Bedrock Integration**: For scalable model hosting and retrieval

---

## ⚙️ How to Run the Application?

### 1️⃣ Create a New Environment

```bash
conda create -n llmapp python=3.8 -y 
```

### 2️⃣ Activate the Environment

```bash
conda activate llmapp 
```

### 3️⃣ Install Required Packages

Install all necessary libraries by running:

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Application

Launch the app with **Streamlit**:

```bash
streamlit run main.py
```

---

## 🐳 Creating a Docker Image

Easily containerize your RAG application using Docker for consistent and reproducible environments!

### 1️⃣ Build the Docker Image

Replace `rag-for-chat` with any name you'd like for your application:

```bash
docker build -t rag-for-chat .
```

### 2️⃣ Verify Image Build

Check your Docker images list to confirm the build:

```bash
docker images
```

### 3️⃣ Run the Docker Image

Run the image with port mapping:

```bash
docker run -p 8083:8083 -it rag-for-chat
```

---

## 🧩 Additional RAG Concepts

- **Retriever**: Fetches relevant documents based on the query, ensuring responses are grounded in the latest or most relevant knowledge.
- **Generator**: Combines retrieved information with language generation for coherent and context-rich responses.

### 📚 Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Amazon Bedrock Documentation](https://aws.amazon.com/bedrock/)
- [Understanding RAG](https://huggingface.co/blog/rag) – A deep dive into Retrieval-Augmented Generation

---

### 🌍 Deploy & Share

Share your RAG application with ease. Containerized with Docker, your app is ready to deploy on any platform supporting Docker containers!

---

Happy building! 🎉
