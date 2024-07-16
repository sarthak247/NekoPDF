"""
Filename: app.py
Description: Implements functions and methods needed for interacting with NekoPDF
Run: streamlit run app.py
"""
import streamlit as st
from dotenv import load_dotenv
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai_chat import get_openai_embeddings, get_openai_answers
from llama_chat import get_llama_embeddings, get_llama_answers

def read_pdf(pdf):
    """
    Parameters:
        - pdf: path to the PDF file
    Return: Returns the contents of the PDF file
    """
    pdf_reader = PdfReader(pdf)

    content = ""
    for page in pdf_reader.pages:
        content += page.extract_text()
    return content

def split_into_chunks(content):
    """
    Parameters:
        - content: the content read from the PDf file
    Return: Returns the contents split into chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200,
            length_function = len
        )
    chunks = text_splitter.split_text(text = content)
    return chunks

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
    pdf = st.file_uploader("Upload your PDF", type = 'pdf')

    # Read PDF
    if pdf is not None:
        # Read PDF content
        content = read_pdf(pdf)

        # Build chunks of text
        chunks = split_into_chunks(content)

        # Accept Questions
        query = st.text_input("Ask questions about your PDF File: ")
        if option == 'GPT 3.5 - Turbo':
            # Check for existing store or create new one
            store_name = pdf.name[:-4] + '.openai.faiss'
            vectorstore = get_openai_embeddings(chunks, store_name)
            if query:
                response = get_openai_answers(vectorstore, query, k)
                st.write(response)
        elif option == 'LLama 2 7B':
            # Check for existing store or create one
            store_name = pdf.name[:-4] + '.llama.faiss'
            vectorstore = get_llama_embeddings(chunks, store_name)
            if query:
                response = get_llama_answers(vectorstore, query, k)
                st.write(response)

if __name__ == '__main__':
    main()
