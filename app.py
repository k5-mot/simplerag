import glob
import mimetypes
import os
import re
import shutil
import statistics
import sys
from typing import List

import chainlit as cl
from chromadb.config import Settings
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.schema import StrOutputParser
from langchain.text_splitter import (
    HTMLHeaderTextSplitter,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
    SpacyTextSplitter,
)
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.document_compressors import FlashrankRerank
from langchain_community.document_loaders import (  # UnstructuredAPIFileLoader,
    PyMuPDFLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredPDFLoader,
)
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_unstructured import UnstructuredLoader

# from operator import itemgetter
# from langchain_core.runnables.config import RunnableConfig
# from langchain_community.vectorstores import FAISS
SEPARATORS = [
    "\n\n", "\n", " ", ".", ",",
    "\u200b",  # Zero-width space
    "\uff0c",  # Fullwidth comma
    "\u3001",  # Ideographic comma
    "\uff0e",  # Fullwidth full stop
    "\u3002",  # Ideographic full stop
    ""
]


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
    """Set file info to metadata for traceability"""
    abspath = os.path.abspath(file_path)
    basename = os.path.basename(file_path)
    for document in documents:
        document.metadata["file_path"] = abspath
        document.metadata["filename"] = basename
        document.metadata["file_type"] = get_filetype(basename)
    return documents


def load_and_split_pdf(file_path: str) -> List[Document]:
    ext = os.path.splitext(file_path)[1]
    if ext != ".pdf":
        return []

    # Load PDF as Document
    loaders = [
        PyMuPDFLoader(file_path=file_path),
        PyPDFLoader(file_path=file_path),
        UnstructuredPDFLoader(file_path=file_path)
    ]
    for loader in loaders:
        try:
            doc = loader.load()
        except Exception as ex:
            print(f"Error loading {file_path}: {ex}")
            continue
        else:
            break

    # Split Document
    text_splitter = SpacyTextSplitter(chunk_size=500, pipeline="ja_core_news_sm")
    splitted_doc = text_splitter.split_documents(doc)
    return splitted_doc


def load_and_split_markdown(file_path: str) -> List[Document]:
    # Load PDF as Document
    loader = TextLoader(file_path=file_path)
    try:
        doc = loader.load()
    except Exception as ex:
        print(f"Error loading {file_path}: {ex}")
        return []

    # Split Document
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
    ]
    text_splitter_md = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        return_each_line=True
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=0,
        length_function=len,
        is_separator_regex=False,
        separators=SEPARATORS
    )
    splitted_docs: List[Document] = []
    for frag in doc:
        splitted_doc = text_splitter_md.split_text(
            frag.page_content.replace("\n\n", "\n")
        )
        splitted_doc = text_splitter.split_documents(splitted_doc)
        splitted_docs.extend(splitted_doc)
    return splitted_doc


def load_and_split_html(file_path: str) -> List[Document]:
    # Load PDF as Document
    loader = TextLoader(file_path=file_path)
    try:
        doc = loader.load()
    except Exception as ex:
        print(f"Error loading {file_path}: {ex}")
        return []

    # Split Document
    headers_to_split_on = [
        ("h1", "Header 1"),
        ("h2", "Header 2"),
    ]
    text_splitter_html = HTMLHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=0,
        length_function=len,
        is_separator_regex=False,
        separators=SEPARATORS
    )
    splitted_docs: List[Document] = []
    for frag in doc:
        splitted_doc = text_splitter_html.split_text(
            frag.page_content.replace("\n\n", "\n")
        )
        splitted_doc = text_splitter.split_documents(splitted_doc)
        splitted_docs.extend(splitted_doc)
    return splitted_doc


def load_and_split_unknown(file_path: str) -> List[Document]:
    # Load PDF as Document
    loader = UnstructuredLoader(
        file_path=file_path,
        chunking_strategy="basic",
        max_characters=1000000,
        include_orig_elements=False
    )
    try:
        doc = loader.load()
    except Exception as ex:
        print(f"Error loading {file_path}: {ex}")
        return []

    # Split Document
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
    ]
    text_splitter_md = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        return_each_line=True
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=0,
        length_function=len,
        is_separator_regex=False,
        separators=SEPARATORS
    )
    splitted_docs: List[Document] = []
    for frag in doc:
        splitted_doc = text_splitter_md.split_text(
            frag.page_content.replace("\n\n", "\n")
        )
        splitted_doc = text_splitter.split_documents(splitted_doc)
        splitted_docs.extend(splitted_doc)
    return splitted_docs


def load_and_split_all(dir_path: str = "/workspace/docs") -> List[Document]:
    """Load various files."""
    # Get files from a specified directory
    file_paths = [
        os.path.abspath(file_path)
        for file_path in glob.glob(f"{dir_path}/**/*", recursive=True)
    ]

    # Convert various file format to plain text
    docs: List[Document] = []
    for file_path in file_paths:
        # Load and Split all files
        ext = os.path.splitext(file_path)[1]
        if ext == ".pdf":
            doc = load_and_split_pdf(file_path)
        elif ext == ".md":
            doc = load_and_split_markdown(file_path)
        elif ext == ".html":
            doc = load_and_split_html(file_path)
        else:
            doc = load_and_split_unknown(file_path)

        # Rearrrange metadatas
        doc = rearrange_metadata(doc)
        doc = add_fileinfo_to_metadata(doc, file_path)
        docs.extend(doc)
        print(f"Loaded {file_path}")
    return docs


def save_documents(docs: List[Document], save_dir: str = "./debug"):
    os.makedirs(save_dir, exist_ok=True)
    for doc in docs:
        filename = doc.metadata["filename"]
        with open(f"{save_dir}/{filename}.txt", mode="a") as f:
            f.write(doc.page_content)
            f.write("\n---\n")
    print(f"Saved Documents to {save_dir}")


def eval_documents(docs: List[Document]):
    docsizes = [len(doc.page_content) for doc in docs]
    print(f"Min: {min(docsizes)}")
    print(f"Max: {max(docsizes)}")
    print(f"Mean: {statistics.mean(docsizes)}")
    print(f"Std: {statistics.stdev(docsizes)}")
    print(f"Var: {statistics.variance(docsizes)}")


def format_docs(docs: List[Document]):
    """Union contexts."""
    return "\n\n".join(doc.page_content for doc in docs)


shutil.rmtree("./debug", ignore_errors=True)

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
# model = ChatAnthropic(temperature=1.0, model_name="claude-3-5-sonnet-20240620")
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
documents = load_and_split_all()
save_documents(documents, "./debug/fin")
eval_documents(documents)

# Dense Retriever
vectorstore.add_documents(documents)
dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Sparse Retriever
sparse_retriever = BM25Retriever.from_documents(documents)
sparse_retriever.k = 4

# Ensemble Retriever
ensemble_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, sparse_retriever],
    weights=[0.5, 0.5]
)

# Reranker
compressor = FlashrankRerank(score_threshold=0.90, top_n=2)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=ensemble_retriever
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
        # print(chunk)
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

        if "id" in md.keys():
            printline = str(md["id"])
        else:
            printline = "?"

        printline += f"({md["file_path"]}): "

        if "relevance_score" in md.keys():
            printline += f"{md["relevance_score"]}"
        print(printline)
        # print(context.page_content)

    # Add documents referenced by RAG to a reply
    for context in contexts:
        # Skip added documents
        context_path = os.path.abspath(context.metadata["file_path"])

        # Add fragment to element
        el_flag = False
        for element in reply.elements:
            if context_path == element.name:
                element.content += "\n- - - - - -\n"
                element.content += context.page_content
                el_flag = True
                continue

        # Add document as element
        if el_flag:
            continue
        reply.elements.append(
            cl.Text(
                name=context_path,
                # language=context.metadata["file_type"],
                content=context.page_content,
                # path=context_path
            )
        )
    await reply.send()
