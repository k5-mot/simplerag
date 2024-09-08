from langchain_community.chat_models import ChatOllama, AzureChatOpenAI
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import SpacyTextSplitter
from langchain_core.vectorstores.base import VectorStore
from langchain_core.documents.base import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
import chainlit as cl
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_chroma import Chroma
import os
from typing import List

SYSTEM_TEMPLATE = "関連ドキュメントを元に、次の質問に答えてください。"
HUMAN_TEMPLATE = """
{question}

以下に、関連ドキュメントを示す。
{context}
"""


@cl.on_chat_start
async def on_chat_start():

    system_prompt = SystemMessagePromptTemplate.from_template(SYSTEM_TEMPLATE)
    human_prompt = HumanMessagePromptTemplate.from_template(HUMAN_TEMPLATE)

    prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
    model = ChatOllama(base_url="http://localhost:30100", model="llama3.1:8b")
    parser = StrOutputParser()
    embeddings = OllamaEmbeddings(
        base_url="http://localhost:30100", model="nomic-embed-text:latest"
    )
    vectorstore = Chroma(embedding_function=embeddings)

    # Get a file from Chainlit UI
    dir_path = "docs/"
    files_file = [
        os.path.abspath(dir_path + f)
        for f in os.listdir(dir_path)
        if os.path.isfile(os.path.join(dir_path, f))
    ]
    print(files_file)

    # Convert various files to texts
    documents = PyMuPDFLoader(file_path=files_file[0]).load()
    # print(file)
    # if file.type == "pdf":
    for f in files_file:
        documents += PyMuPDFLoader(file_path=f).load()

    # Split texts for token reduction
    text_splitter = SpacyTextSplitter(chunk_size=400, pipeline="ja_core_news_sm")
    splitted_documents = text_splitter.split_documents(documents)
    vectorstore.add_documents(splitted_documents)

    runnable = (
        {"context": vectorstore.as_retriever(), "question": RunnablePassthrough()}
        | prompt
        | model
        | parser
    )

    cl.user_session.set("data", vectorstore)
    cl.user_session.set("runnable", runnable)


@cl.on_message
async def on_message(message: cl.Message):
    runnable: Runnable = cl.user_session.get("runnable")
    vectorstore: VectorStore = cl.user_session.get("data")
    documents = vectorstore.similarity_search(message.content)
    context = ""
    for document in documents:
        context += f"""
        ```
        {document.page_content}
        ```
        """

    msg = cl.Message(content="")
    async for chunk in runnable.astream(
        {"question": message.content, "context": context},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()
