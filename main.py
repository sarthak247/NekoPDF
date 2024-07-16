from fastapi import FastAPI
from pydantic import BaseModel
from llama_chat import get_llama_embeddings, get_llama_answers
from typing import List
from openai_chat import get_openai_answers, get_openai_embeddings

app = FastAPI()

class QA(BaseModel):
    chunks : List
    store_name : str
    query : str
    k: int

@app.post('/qa/openai')
def openai_response(input: QA):
    vectorstore = get_openai_embeddings(input.chunks, input.store_name)
    if input.query:
        response = get_openai_answers(vectorstore, input.query, input.k)
        return response
    
@app.post('/qa/llama')
def llama_response(input: QA):
    vectorstore = get_llama_embeddings(input.chunks, input.store_name)
    if input.query:
        response = get_llama_answers(vectorstore, input.query, input.k)
        return response

    