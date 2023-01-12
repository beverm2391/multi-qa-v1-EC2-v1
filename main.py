from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import time

# ! Get Embeddings Imports
from sentence_transformers import SentenceTransformer
from torch import save
import torch

# ! Get Tokens Imports
from transformers import GPT2Tokenizer

# ! START CONFIG -------------------------------------

app = FastAPI()

origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ? Tokenizer
try:
    tokenizer = torch.load("tokenizer.pt")
    print("Tokenizer Successfully Loaded")
except:
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    print("Tokenizer Downloaded")
    torch.save(tokenizer, "tokenizer.pt")

# ? Embeddings Model
try:
    model = torch.load("model.pt")
    print("Model Successfully Loaded")
except:
    model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')
    print("Model Downloaded")
    torch.save(model, "model.pt")

# ! END CONFIG -------------------------------------
# ! START CLASSES -------------------------------------
class Text(BaseModel):
    text: str

# ! END CLASSES -------------------------------------
# ! START FUNCTIONS -------------------------------------
# ? Generate Embeddings
def get_embedding(data: str, model) -> List[float]:
    print("Generating Embedding")
    embedding = model.encode(data)
    print("Embedding Successful")
    return embedding

# ? Get Number of Tokens
def get_tokens(text: str, tokenizer):
    start = time.perf_counter()
    tokens = tokenizer.tokenize(text)
    elapsed = time.perf_counter() - start
    return (len(tokens), elapsed)

# ! END FUNCTIONS -------------------------------------
# ! START ROUTES -------------------------------------
@app.get("/")
def root():
    return {"message": "API is up and running."}

@app.get("/embeddings")
def route_embedding_info():
    return {"message": "This endpoint expects a POST request with a JSON body containing a text field. It returns a vector embedding of dimensions (768,)."}

@app.post("/embeddings")
def handle_embedding(data: Text):
    embedding = get_embedding(data.text, model)
    response_embedding = embedding.tolist()
    return {"embedding": response_embedding}

@app.get("/tokens")
def route_tokens_info():
    return {"message": "This endpoint expects a POST request with a JSON body containing a text field. It returns the number of tokens."}

@app.post("/tokens")
def handle_tokens(data: Text):
    tokens = get_tokens(data.text, tokenizer)
    token_len = len(tokens)
    return {"tokens": token_len}