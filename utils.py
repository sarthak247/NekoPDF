"""
Filename: utils.py
Description: Implements functions and methods needed for reading text from files
                and splitting into chunks, etc
"""
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def read_pdf(pdfs):
    """
    Parameters:
        - pdf: path to PDF files
    Return: Returns the contents of the PDF file
    """
    content = ""
    for pdf in pdfs:
        pdf_reader = PdfReader(pdf)
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

def get_store_name(pdfs):
    """
    Construct a unique store name for the uploaded PDF files. We sort the files ensuring
    that the store name remains same for the same files
    Parameters:
        pdf: path to PDF files
    Return: sorted store name
    """
    pdfs = sorted(pdfs, key = lambda file: file.name)
    store_name = ''
    for pdf in pdfs:
        store_name += (pdf.name[:-4] + '.')
    return store_name
