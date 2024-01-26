import os

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import OpenAI, OpenAIEmbeddings

load_dotenv()
openai_api_key = os.getenv('openai_api_key')

# Define the list of documents as strings
documents = [
    Document(page_content="John is 25 years old"),
    Document(page_content="Football is called soccer in Europe"),
    Document(page_content="The hosts are Rached Bouchoucha and Bastien Decorte"),
    Document(page_content="Python? Java? Which to learn first?"),
    Document(page_content="Minecraft's an awesome game"),
]

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
vector_store = Chroma(embedding_function=embeddings)
vector_store.add_documents(documents)

retriever = VectorStoreRetriever(
    vectorstore=vector_store,
    search_kwargs={"k": 3, "score_threshold": 0.7},
    search_type="similarity_score_threshold",
)
retrievalQA = RetrievalQA.from_llm(llm=OpenAI(openai_api_key=openai_api_key), retriever=retriever)

answer = retrievalQA.invoke({"query": "Who are the hosts?"})
print("1. The answer is:", answer["result"].strip(), "\n")

answer = retrievalQA.invoke({"query": "which game is good?"})
print("2. The answer is:", answer["result"].strip(), "\n")