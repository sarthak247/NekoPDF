"""
Filename: app.py
Description: Implements functions and methods needed for interacting with NekoPDF
Run: streamlit run app.py
"""
import json
import streamlit as st
from dotenv import load_dotenv
from streamlit_extras.add_vertical_space import add_vertical_space
import requests
from utils import read_pdf, split_into_chunks, get_store_name

# Favicon and Title
st.set_page_config(page_title="NekoPDF 📖 - Chat with PDF",
                   page_icon="🐱", layout="centered",
                   initial_sidebar_state="auto",
                   menu_items=None)

# SideBar
with st.sidebar:
    st.title("🐱 NekoPDF 📖 - Chat with PDF")
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io)
    - [LangChain](https://python.langchain.com)
    ''')
    add_vertical_space(4)
    st.write("Made with :sparkling_heart: by [Sarthak Thakur](https://sarthak247.github.io)")

def main():
    # Load Environment Variables
    load_dotenv()

    # Main App
    st.header("🐱 NekoPDF - Chat with PDF 📖")

    # Select LLM
    option = st.selectbox('Select LLM', ('GPT 3.5 - Turbo', 'LLama 2 7B'))

    # Select top-k similarity search
    k = st.slider('Top K', 1, 5, 1)

    # Upload PDF File
    pdfs = st.file_uploader("Upload your PDF", type = 'pdf', accept_multiple_files=True)
    store_name = get_store_name(pdfs)

    # Read PDF
    if pdfs is not None:
        # Read PDF content
        content = read_pdf(pdfs)

        # Build chunks of text
        chunks = split_into_chunks(content)

        # Accept Questions
        query = st.text_input("Ask questions about your PDF File: ")
        if option == 'GPT 3.5 - Turbo':
            # Check for existing store or create new one
            store_name =store_name + 'openai.faiss'
            payload = {'chunks': chunks, 'store_name' : store_name, 'query' : query, 'k': k}
            if query:
                response = requests.post(url='http://127.0.0.1:8000/qa/openai',
                                         data = json.dumps(payload),
                                         timeout=120)
                st.write(response.text)
        elif option == 'LLama 2 7B':
            # Check for existing store or create one
            store_name = store_name + 'llama.faiss'
            payload = {'chunks' : chunks, 'store_name' : store_name, 'query' : query, 'k' : k}
            if query:
                response = requests.post(url='http://127.0.0.1:8000/qa/llama',
                                         data = json.dumps(payload),
                                         timeout=120)
                st.write(response.text)

if __name__ == '__main__':
    main()
