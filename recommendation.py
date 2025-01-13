import openai
import pinecone
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with specific origins if required for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Keys
openai.api_key = "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"  # Replace with your OpenAI API key
PINECONE_API_KEY = "pcsk_3ZyWhu_AyLNPgS9ZaerfM7ysrBVHmzVfwdzxDDfp1mRzuisKTUPtnZjYARPvHBpY3Vn45q"  # Replace with your Pinecone API key
PINECONE_ENV = "us-west1-gcp"  # Replace with your Pinecone environment

# Pinecone Initialization
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
vector_db_name = "rag-recommendation-db"
if vector_db_name not in pinecone.list_indexes():
    pinecone.create_index(vector_db_name, dimension=1536)  # Dimension for `text-embedding-ada-002`
vector_db = pinecone.Index(vector_db_name)

# Data Models
class AddDataRequest(BaseModel):
    data: str

class RecommendationRequest(BaseModel):
    query: str

class SentimentAnalysisRequest(BaseModel):
    text: str

# Utility Functions
def chunk_data(data: str, chunk_size: int = 512) -> List[str]:
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

def store_embeddings(data_chunks: List[str], metadata: Dict = None):
    for i, chunk in enumerate(data_chunks):
        response = openai.Embedding.create(input=chunk, model="text-embedding-ada-002")
        embedding = response["data"][0]["embedding"]
        vector_db.upsert([{
            "id": f"chunk-{i}",
            "values": embedding,
            "metadata": metadata or {"text": chunk}
        }])

def retrieve_documents(query: str, top_k: int = 5) -> List[Dict]:
    query_embedding = openai.Embedding.create(input=query, model="text-embedding-ada-002")["data"][0]["embedding"]
    response = vector_db.query(query_embedding, top_k=top_k, include_metadata=True)
    return response.get("matches", [])

def generate_rag_response(query: str, documents: List[Dict]) -> str:
    context = "\n".join([f"- {doc['metadata']['text']}" for doc in documents])
    prompt = f"""
You are a highly intelligent assistant. Use the following retrieved documents to answer the query:
Query: {query}
Retrieved Documents:
{context}

Provide a detailed and accurate response based on the retrieved information.
"""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"]

# Endpoints
@app.post("/api/add-data")
async def add_data(request: AddDataRequest):
    try:
        chunks = chunk_data(request.data)
        store_embeddings(chunks)
        return {"message": "Data added successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/recommendation")
async def recommendation(request: RecommendationRequest):
    try:
        documents = retrieve_documents(request.query)
        response = generate_rag_response(request.query, documents)
        return {"query": request.query, "recommendations": response, "retrieved_documents": documents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/sentiment")
async def sentiment_analysis(request: SentimentAnalysisRequest):
    try:
        prompt = f"Analyze the sentiment of the following text: {request.text}"
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return {"text": request.text, "sentiment_analysis": response["choices"][0]["message"]["content"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
