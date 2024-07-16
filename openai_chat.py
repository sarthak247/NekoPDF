"""
Filename: openai.py
Description: Implements functions needed to work around with OpenAI API
"""
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.callbacks.manager import get_openai_callback

def get_openai_embeddings(chunks, store_name):
    """
    Parameters:
        - chunks: text to turn into embeddings
        - store_name : The name of the store from which to load in
                        case of existing embeddings or create and save to
    Return: An instance of FAISS Vectorstore
    """
    embeddings = OpenAIEmbeddings()
    if os.path.exists(store_name):
        vectorstore = FAISS.load_local(store_name, embeddings, allow_dangerous_deserialization=True)
    else:
        # Convert chunks -> Embeddings
        vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
        vectorstore.save_local(store_name)
    return vectorstore


def get_openai_answers(vectorstore, query, k):
    """
    Parameters:
        - vectorstore: Vector Store of chunks of texts and their embeddings
        - Query: Question to ask to the LLM
        - k: Number of top k matching documents from similarity search
    Return: Response from OpenAI API
    """
    docs = vectorstore.similarity_search(query, k)
    # Setup LLM
    llm = ChatOpenAI(temperature=0, model_name = "gpt-3.5-turbo")

    # Setup QA Chain and query it
    chain = load_qa_chain(llm = llm, chain_type = "stuff")
    input_data = {'input_documents' : docs, 'question' : query}
    with get_openai_callback() as cb:
        response = chain.invoke(input=input_data)
        print(cb)
    return response['output_text']
