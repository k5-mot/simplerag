import glob
import mimetypes
import os
import re
from typing import List

import chainlit as cl
from chromadb.config import Settings
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.retrievers import ContextualCompressionRetriever
from langchain.schema import StrOutputParser
from langchain.text_splitter import SpacyTextSplitter
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.document_compressors import FlashrankRerank
from langchain_community.document_loaders import (  # UnstructuredAPIFileLoader,
    PyMuPDFLoader,
)
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_unstructured import UnstructuredLoader

# from operator import itemgetter
# from langchain_core.runnables.config import RunnableConfig
# from langchain_community.vectorstores import FAISS


def rearrange_metadata(documents: List[Document]) -> List[Document]:
    """Rearrange metadata for vector stores"""
    for document in documents:
        # Remove unsupported datatypes by vector stores
        for k, v in document.metadata.items():
            metatype = type(v)
            if metatype in (str, int, float, bool):
                continue
            elif metatype in (list, tuple):
                document.metadata[k] = document.metadata[k][0]
            else:
                document.metadata.pop(k)
    return documents


def get_filetype(filepath: str) -> str:
    mimetypes.add_type("text/x-toml", ".toml")
    mimetypes.add_type("text/x-yaml", ".yml")
    mimetypes.add_type("text/x-yaml", ".yaml")
    mimetypes.add_type("text/x-sh", ".gitignore")
    guess_type = mimetypes.guess_type(filepath)[0] or ""
    mime_type = re.sub("^[a-z]+/", "", guess_type)
    mime_type = re.sub("^x-", "", mime_type)
    mime_type = re.sub("^plain", "text", mime_type)
    return mime_type.lower()


def add_fileinfo_to_metadata(
    documents: List[Document], file_path: str
) -> List[Document]:
    """Set file info to metadata"""
    abspath = os.path.abspath(file_path)
    basename = os.path.basename(file_path)
    for document in documents:
        document.metadata["file_path"] = abspath
        document.metadata["filename"] = basename
        document.metadata["file_type"] = get_filetype(basename)
    return documents


def load_multifiles(documents_dir: str = "docs") -> List[Document]:
    """Load various files."""
    # Get files from a specified directory
    filepaths = [
        os.path.abspath(filepath)
        for filepath in glob.glob(f"./{documents_dir}/**/*", recursive=True)
    ]

    # Convert various file format to plain text
    documents: List[Document] = []
    for filepath in filepaths:
        # Setup document loader
        ext = os.path.splitext(filepath)[1]
        if ext == ".pdf":
            loader = PyMuPDFLoader(file_path=filepath)
        else:
            loader = UnstructuredLoader(file_path=filepath)

        # Load documents
        try:
            document = loader.load()
        except Exception as ex:
            print(f"Error loading {filepath}: {ex}")
            continue
        print(f"Loaded {filepath}")
        documents.extend(add_fileinfo_to_metadata(document, filepath))
    return rearrange_metadata(documents)


def format_docs(docs: List[Document]):
    """Union contexts."""
    return "\n\n".join(doc.page_content for doc in docs)


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
model = ChatAnthropic(temperature=1.0, model_name="claude-3-5-sonnet-20240620")
model = ChatOllama(base_url=OLLAMA_API_URL, model="llama3.1:8b")

# Output parser
parser = StrOutputParser()

# Embedding model
embeddings = OllamaEmbeddings(base_url=OLLAMA_API_URL, model="nomic-embed-text:latest")

# Vector store
# vectorstore = FAISS.from_documents(embedding=embeddings, documents=[Document("")])
vectorstore = Chroma(
    embedding_function=embeddings, client_settings=Settings(anonymized_telemetry=False)
)

# Documents loader
documents = load_multifiles()

# Text Splitter
text_splitter = SpacyTextSplitter(chunk_size=300, pipeline="ja_core_news_sm")
splitted_documents = text_splitter.split_documents(documents)

# Retriever
vectorstore.add_documents(splitted_documents)
retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

# Reranker
compressor = FlashrankRerank(score_threshold=0.90, top_n=3)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

# RAG chain
rag_chain = (
    RunnablePassthrough(context=(lambda x: format_docs(x["context"])))
    | prompt
    | model
    | parser
)

# Rerank RAG chain
runnable = RunnableParallel(
    {"context": compression_retriever, "question": RunnablePassthrough()}
).assign(answer=rag_chain)


@cl.on_message
async def on_message(message: cl.Message):
    reply = cl.Message(content="", elements=[])

    # Stream a reply message
    async for chunk in runnable.astream(message.content):
        if "answer" in chunk.keys():
            await reply.stream_token(chunk["answer"])
        elif "context" in chunk.keys():
            contexts: List[Document] = chunk["context"]
        elif "question" in chunk.keys():
            question: str = chunk["question"]
    await reply.send()

    # Print intermediates
    print(question)
    for context in contexts:
        md = context.metadata
        print(f"{md["id"]} ({md["file_path"]}): {md["relevance_score"]}")

    # Add documents referenced by RAG to a reply
    for context in contexts:
        # Skip added documents
        context_path = os.path.abspath(context.metadata["file_path"])
        if any(context_path == element.name for element in reply.elements):
            continue
        # Add document as element
        reply.elements.append(
            cl.Text(
                name=context_path,
                # language=context.metadata["file_type"],
                content=context.page_content,
                # path=context_path
            )
        )
    await reply.send()
