from src.helper import load_documents, text_splitter, download_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

extracted_data = load_documents("data/")

chunks = text_splitter(extracted_data)

embeddings = download_embeddings()

#create pinecone index
pc = Pinecone(api_key=PINECONE_API_KEY)

pc.create_index(
  name="medibot",
  dimension=384,
  metric="cosine",
  spec=ServerlessSpec(
    cloud="aws",
    region="us-east-1"
  )
)

#embed each chunk and store in pinecone index
docsearch = PineconeVectorStore.from_documents(
    documents=chunks,
    index_name="medibot",
    embedding=embeddings
)