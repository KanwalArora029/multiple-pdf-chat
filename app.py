import streamlit as st
from dotenv import load_dotenv
import os
from pyPDF2 import PdfReader


def get_pdf_text(pdf_docs):
    pdf_text = ""
    for pdf_doc in pdf_docs:
        pdf_reader = PdfReader(pdf_doc)
        for page in pdf_reader.pages:
            pdf_text += page.extractText()        
    return pdf_text


def main():
    load_dotenv()
    st.set_page_config(page_title='Chat  with multiple PDF', page_icon=':books')

    st.title('Chat  with multiple PDF')
    st.text_input("Ask a questtion about your documents:")

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", 
            accept_multiple_files=True    
        )
        if st.button("Process"):
            with st.spinner("Processing"):
                ## get PDF text
                raw_text = get_pdf_text(pdf_docs)

                ## get text chunks

                ## create vector store

if __name__=='__main__':
    main()