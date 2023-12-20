# Demo App: Building AI Applications with ChatGPT and OpenAI LLMs
# by Sumudu Tennakoon
# Last Update: 2022-07-09

import streamlit as st
import configparser
import traceback
import os
import tempfile
import pandas as pd
import datetime
from openai import OpenAI
import chromadb

from langchain.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Get OpenAI Key
config = configparser.ConfigParser()
config.read('config.ini')
openai_api_key = config['SECRETS']['openai_api_key']

MODEL = "gpt-3.5-turbo"

# LLM Client
client = OpenAI(api_key=openai_api_key)

# Embeding Function
embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={'device': 'cuda'},
)

st.set_page_config (layout="wide")
st.title("📖 Document Q&A")

if "uploaded_file" not in st.session_state:
    st.session_state["uploaded_file"] = None

if "retriever" not in st.session_state:
    st.session_state["retriever"] = None


def read_document_file_pages_to_documents(file_path):
    if os.path.exists(file_path):
        loader = UnstructuredPDFLoader(file_path, mode='paged')
        pages_docs = loader.load()
    else:
        pages_docs = []

    return pages_docs

def read_document_file_pages_to_df(file_path, metadata={}, content_field_name ='page_content'):
    pages_docs = read_document_file_pages_to_documents(file_path)
    
    pages_docs_df = []
    for doc in pages_docs:
        try:
            page_content = doc.page_content
            doc = doc.metadata
            doc[content_field_name] = page_content

            # Overrides metadata from custom fields values
            if "file_name" in metadata:
                doc['file_name'] = metadata['file_name']
            if "uri" in metadata:
                doc['uri'] = metadata['uri']
                
            pages_docs_df.append(doc)
        except:
            print(traceback.format_exc())

    return pd.DataFrame(pages_docs_df)

def convert_uploaded_document_pages_to_df(uploaded_file, metadata={}):
    file_name = uploaded_file.name
    metadata["file_name"] = file_name
    if uploaded_file.type == 'application/pdf':

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:            
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        pages_df = read_document_file_pages_to_df(temp_file_path, metadata, content_field_name ='page_content')
        
        #selete file
        os.remove(temp_file_path) 
    else:
        pages_df = pd.DataFrame()
        
    return file_name, pages_df


def create_retriever_from_pages_df(pages_df, embedding_function, collection_name='doc_qa', columns_to_retriever=['file_name', "page_content", 'page_number'], chunk_size=300, chunk_overlap=0, k=3):

    # DataFrame to Documents
    loader = DataFrameLoader(pages_df[columns_to_retriever], page_content_column="page_content")
    page_docs = loader.load()

    # Split into Chunks  
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, 
                                                chunk_overlap=chunk_overlap,
                                                separators=['\n\n', '\n', '.', ' ', ''],
                                                )
    page_chunks = text_splitter.split_documents(page_docs)

    try:
        chromadb.Client().delete_collection('doc_qa')
    except:
        pass

    retriever = Chroma.from_documents(page_chunks, embedding_function, collection_name=collection_name).as_retriever(search_kwargs={"k": k})

    return retriever

def doc_qna(question, client, retriever):
    
    MODEL = "gpt-3.5-turbo"
    
    INSTRUCTIONS = """Your primary function in this interaction is to provide information and answer questions based on the document context provided to you. 
    You should not generate responses based on information outside of the given document context. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    If answer contain a list, output as a bulleted or numbered list. If answer contain a table return as tab delimited.
    """

    question = question.strip()

    # Context Retrival
    fetched_docs = retriever.get_relevant_documents(question)
    #st.write(fetched_docs)
    fetched_docs_df = pd.DataFrame([{"file_name": doc.metadata["file_name"], "page_content":doc.page_content, "page_number":doc.metadata["page_number"]} for doc in fetched_docs])
    context = 'Context:\n----------------\n'+'\n\n----------------\\n'.join(fetched_docs_df.sort_values(by='page_number')['page_content'].values)
    
    # Assembling System Prompt
    system_prompt = F"{INSTRUCTIONS}\n\nContext:{context}"

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        temperature=0,
        max_tokens=500
    )
    
    answer = response.choices[0].message.content
    usage = response.usage.model_dump()

    return answer, fetched_docs_df.to_dict(orient="records"), usage

def create_retriver():
    file_name, pages_df = convert_uploaded_document_pages_to_df(st.session_state["uploaded_file"], metadata={})

    with st.sidebar:
        st.write(F"File: {file_name}")
        st.warning(F"Last refreshed time {datetime.datetime.now()}")
    #st.dataframe(pages_df)

    st.session_state["retriever"] = create_retriever_from_pages_df(pages_df, embedding_function)

# Upload Document
with st.sidebar:
    st.subheader("Document Upload")
    st.session_state["uploaded_file"] = st.file_uploader("Browse document and click the Submit button below.", type="pdf")
    btn_create_retriver = st.button("Submit Document to AI Reader", type="primary", use_container_width=True)

    if btn_create_retriver:
        create_retriver()

# Text Field to Ask Question
question = st.text_input(
    "Ask something from the article",
    placeholder="Can you give me a short summary?",
    disabled=not st.session_state["uploaded_file"],
    help="upload document to enable question input"
)

if question and st.session_state["retriever"]:
    answer, sources, usage  = doc_qna(question, client, st.session_state["retriever"] )
    st.success(answer)
    st.info(usage)
    st.subheader("Sources:")
    st.write(sources)