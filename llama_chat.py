"""
Filename: openai.py
Description: Implements functions needed to work around with Llama.cpp for QA
"""
import os
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate

# Setup LLM
llm = LlamaCpp(model_path = './models/llama-2-7b-chat.Q2_K.gguf',
               temperature = 0.75,
               max_tokens = 2000,
               top_p = 1,
               verbose = False,
               n_gpu_layers = -1,
               n_batch = 128,
               n_ctx = 1024)

embeddings = LlamaCppEmbeddings(model_path = './models/llama-2-7b-chat.Q2_K.gguf',
                                    n_gpu_layers = -1, verbose = False)

# Sample Template
TEMPLATE = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
{context}
Question: {question}
Answer:"""
prompt = PromptTemplate.from_template(TEMPLATE)

def get_llama_embeddings(chunks, store_name):
    """
    Parameters:
        - chunks: text to turn into embeddings
        - store_name : The name of the store from which to load in
                        case of existing embeddings or create and save to
    Return: An instance of FAISS Vectorstore
    """
    if os.path.exists(store_name):
        vectorstore = FAISS.load_local(store_name, embeddings, allow_dangerous_deserialization=True)
    else:
        # Convert chunks -> Embeddings
        vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
        vectorstore.save_local(store_name)
    return vectorstore

def get_llama_answers(vectorstore, query, k):
    """
    Parameters:
        - vectorstore: Vector Store of chunks of texts and their embeddings
        - Query: Question to ask to the LLM
        - k: Number of top k matching documents from similarity search
    Return: Response from llama model
    """
    docs = vectorstore.similarity_search(query, k)

    # Extract context
    context = ''
    for doc in docs:
        context += doc.page_content

    # Setup chain
    llm_chain = prompt | llm
    response = llm_chain.invoke({'context' : context, 'question' : query})
    return response
