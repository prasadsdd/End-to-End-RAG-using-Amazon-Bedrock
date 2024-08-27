# End-to-End-RAG-using-Amazon-Bedrock
![Screenshot (11)](https://github.com/user-attachments/assets/51a265f2-6318-4eb5-a958-026d49a12853)

### How to run?

##  1.Create a new environment

```bash
conda create -n llmapp python=3.8 -y 
```


##  2.Activate the environment
```bash
conda activate llmapp 
```



##  3.Install the requirements package
```bash
pip install -r requirements.txt
```


##  4. run your application

```bash

streamlit run main.py

```

###  Creating Docker Image

## 1.Build Image
```bash
docker build -t rag-for-chat . (rag-for-chat is the application name u can put any name you want)
```

## 2.Check Image list
```bash
docker images
```

## 3.Run the Docker Image
```bash
docker run -p 8083:8083 -it rag-for-chat
```
