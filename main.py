import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import bs4

from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv(dotenv_path=".venv/.env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")

if LANGSMITH_TRACING:
    os.environ["LANGSMITH_TRACING"] = LANGSMITH_TRACING
    os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY


if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Passo 1: Instanciando o Modelo LLM
    llm = init_chat_model("llama3-8b-8192", model_provider="groq")
else:
    print("GROQ_API_KEY não encontrada no arquivo .env")


# Passo 2: Criando o embedding Model

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Passo 3:  Criando o Vector Store

# Criar o vector store Chroma
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

# Passo 4: Aplicação

# Load and chunk contents of the blog
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# Index chunks
_ = vector_store.add_documents(documents=all_splits)

# Define prompt for question-answering
prompt = hub.pull("rlm/rag-prompt")


# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

response = graph.invoke({"question": "What is self-reflection?, answer in portuguese BR"})
print(response["answer"])
