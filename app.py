import glob
import os

import chainlit as cl
from chromadb.config import Settings
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import StrOutputParser
from langchain.schema.runnable.config import RunnableConfig
from langchain.text_splitter import SpacyTextSplitter
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import (  # UnstructuredAPIFileLoader,
    PyMuPDFLoader,
)
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_unstructured import UnstructuredLoader

SYSTEM_TEMPLATE = "関連ドキュメントを元に、次の質問に答えてください。"
HUMAN_TEMPLATE = """
{question}

以下に、関連ドキュメントを示す。
{context}
"""
OLLAMA_API_URL = "http://localhost:30100"
OLLAMA_API_URL = "http://host.docker.internal:30100"

# Prompt
system_prompt = SystemMessagePromptTemplate.from_template(SYSTEM_TEMPLATE)
human_prompt = HumanMessagePromptTemplate.from_template(HUMAN_TEMPLATE)
prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

# LLM
model = ChatOllama(base_url=OLLAMA_API_URL, model="llama3.1:8b")

# Output parser
parser = StrOutputParser()

# Embedding model
embeddings = OllamaEmbeddings(base_url=OLLAMA_API_URL, model="nomic-embed-text:latest")

# Vector store
vectorstore = Chroma(
    embedding_function=embeddings, client_settings=Settings(anonymized_telemetry=False)
)

# Get a file from Chainlit UI
documents_dir = "docs"
documents_path = glob.glob(f"./{documents_dir}/**/*", recursive=True)

# Convert various files to texts
documents = []
for document_path in documents_path:
    try:
        document_path = os.path.abspath(document_path)
        print(document_path)

        ext = os.path.splitext(document_path)[1]
        if ext == ".pdf":
            loader = PyMuPDFLoader(file_path=document_path)
        else:
            loader = UnstructuredLoader(file_path=document_path)
        document = loader.load()
        for doc in document:
            for k, v in doc.metadata.items():
                metatype = type(v)
                if metatype in (str, int, float, bool):
                    continue
                elif metatype in (list, tuple):
                    doc.metadata[k] = doc.metadata[k][0]
                else:
                    doc.metadata.pop(k)
            if "filename" not in doc.metadata.keys():
                doc.metadata["filename"] = document_path

        documents += document
    except Exception as ex:
        print(f"Error loading {document_path}: {ex}")

# Split texts for token reduction
text_splitter = SpacyTextSplitter(chunk_size=400, pipeline="ja_core_news_sm")
splitted_documents = text_splitter.split_documents(documents)
vectorstore.add_documents(splitted_documents)

# RAG chain
runnable = (
    {"context": vectorstore.as_retriever(), "question": RunnablePassthrough()}
    | prompt
    | model
    | parser
)


@cl.on_message
async def on_message(message: cl.Message):
    reply = cl.Message(content="")
    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await reply.stream_token(chunk)
    await reply.send()
