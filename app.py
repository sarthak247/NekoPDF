import streamlit as st
from dotenv import load_dotenv
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import pickle
import os


# Favicon and Title
st.set_page_config(page_title="NekoPDF 📖 - Chat with PDF", page_icon="🐱", layout="centered", initial_sidebar_state="auto", menu_items=None)

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

    # Upload PDF File
    pdf = st.file_uploader("Upload your PDF", type = 'pdf')
    
    # Read PDF
    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        content = ""
        for page in pdf_reader.pages:
            content += page.extract_text()

        # Build chunks of text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200,
            length_function = len
        )

        chunks = text_splitter.split_text(text = content)

        # Check for existing store or create new one
        store_name = pdf.name[:-4]
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
        else:
            # Convert chunks -> Embeddings
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)

            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)
        
        # Accept Questions
        query = st.text_input("Ask questions about your PDF File: ")
        if query:
            docs = VectorStore.similarity_search(query = query, k = 3)
            
            # Setup LLM
            llm = OpenAI(temperature=0, model_name = "gpt-3.5-turbo")

            # Setup QA Chain and query it
            chain = load_qa_chain(llm = llm, chain_type = "stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents = docs, question = query)
                print(cb)
            st.write(response)



    

if __name__ == '__main__':
    main()