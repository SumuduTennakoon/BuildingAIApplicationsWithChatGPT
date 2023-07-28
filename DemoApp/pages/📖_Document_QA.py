# Demo App: Building AI Applications with ChatGPT and OpenAI LLMs
# by Sumudu Tennakoon
# Last Update: 2022-07-09

import streamlit as st
import configparser
import openai
import os

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Get OpenAI Key
config = configparser.ConfigParser()
config.read('config.ini')
openai_api_key = config['SECRETS']['openai_api_key']

prompt_template = """Use the following pieces of CONTEXT to answer the question at the end. \
Do not answer anything outside the CONTEXT given. If you don't know the answer, just say that you don't know, don't try to make up an answer.\
If answer contain a list, output as a bulleted or numbered list. If answer contain a table return as tab delimited.

CONTEXT: {context}

Question: {question}
Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

chain_type_kwargs = {"prompt": PROMPT}

MODEL = "gpt-3.5-turbo"

chat_llm = ChatOpenAI(model=MODEL, 
                    temperature=0, 
                    max_tokens=500, 
                    openai_api_key=openai_api_key,
                    verbose=True)

st.title("📖 Document Q&A")

# Upload Document
uploaded_file = st.file_uploader("Upload an article", type="pdf")

if uploaded_file is not None:
    output_temp_file_path = os.path.join("temp", uploaded_file.name)
    with open(output_temp_file_path, 'wb') as output_temp_file:
        output_temp_file.write(uploaded_file.read())

    loader = PyPDFLoader(output_temp_file_path)
    pages = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, 
                                                chunk_overlap=10,
                                                length_function = len,
                                                separators=['\n\n', '\n', '.', ' ', ''],
                                                add_start_index = True,
                                                )

    text_chunks = text_splitter.split_documents(pages)

    # Create Knowladge Store and Retriver
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cuda'},
    )

    retriever = Chroma.from_documents(text_chunks, embeddings)

    # RetrievalQA Chain
    qa = RetrievalQA.from_chain_type(llm=chat_llm, 
                                    chain_type="stuff", 
                                    retriever=retriever.as_retriever(), 
                                    chain_type_kwargs=chain_type_kwargs, 
                                    return_source_documents=True)

    # Text Field to Ask Question
    question = st.text_input(
        "Ask something about the article",
        placeholder="Can you give me a short summary?",
        disabled=not uploaded_file,
    )

    # Answer Question and Return with Sources
    if question and openai_api_key:
        query = "why we need responsible approach to AI?"
        result = qa({"query": question})
        st.success(result['result'])
        st.write(result['source_documents'])