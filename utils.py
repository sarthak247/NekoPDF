"""
Filename: utils.py
Description: Implements functions and methods needed for reading text from files
                and splitting into chunks, etc
"""
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

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