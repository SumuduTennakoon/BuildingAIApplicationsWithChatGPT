# Demo App: Building AI Applications with ChatGPT and OpenAI LLMs
# by Sumudu Tennakoon
# Last Update: 2022-07-09

import openai
from openai import OpenAI
import streamlit as st
import configparser
import os

from langchain.chains import create_sql_query_chain
from langchain.schema import HumanMessage, SystemMessage
from langchain.chat_models import ChatOpenAI
from langchain.llms.openai import OpenAI
from langchain.sql_database import SQLDatabase

from PIL import Image


# Get OpenAI Key
config = configparser.ConfigParser()
config.read('config.ini') #Change to your path or assign API Key to openai_api_key (not recomended for production)
openai_api_key = config['SECRETS']['openai_api_key']

MODEL = "gpt-3.5-turbo"

llm_client = ChatOpenAI(api_key=openai_api_key, model=MODEL, verbose=True)

db_uri = r"sqlite:///D:/Data/Chinook.db"
db = SQLDatabase.from_uri(db_uri)

st.set_page_config (layout="wide")
st.title("↔💻 SQL Data Query Assistant")
with st.sidebar:
    st.subheader("Sample Questions")
    st.markdown("""
    * How many employees are there?
    * How many invoices with amount greater than 10?
    * Who is the customer who spent the most and the amount?            
    * List the total sales per country
    * Which country's customers spent the most and the amount?
    * List all tracks that are longer than 5 minutes?
    * List all customers from Canada?
    """)

st.write(r'https://github.com/lerocha/chinook-database')
image = Image.open(r'D:\Data\db_diagram.jpg') 
st.image(image) 

def get_sql_query(chain, question):    
    query = chain.invoke({"question": question})
    return query

def execute_sql_query(db, query):
    result = str(db.run(query))
    if result=="":
        return None
    else:
        return result

def get_answer(llm_client, db, question):
    chain = create_sql_query_chain(llm_client, db)
    query = get_sql_query(chain, question)
    result = execute_sql_query(db, query)

    INSTRUCTIONS = F"""You are a Data Analyst Chatbot. You are given a SQLQuery writen to ftech data to answer the user question and the Result.
    Your task is provide natural language answer to user question. If you did not find values in Result, reply with "Sorry I did not find records to answwer your question."

    SQLQuery: {query}

    Result: {result}
    """

    messages = [
        SystemMessage(
            content=INSTRUCTIONS
        ),
        HumanMessage(
            content=question
        ),
    ]
    
    response = llm_client(
        model=MODEL,
        messages=messages,
        temperature=0,
        max_tokens=1000,
    )
    answer = response.content

    print(response)

    return {"query":query, "result":result, "answer":answer}

with st.form("my_form"):
    question = st.text_input("User Question:", "")
    submitted = st.form_submit_button("Submit")
    if submitted:
        response = get_answer(llm_client, db, question)
        st.success(response["answer"])
        with st.expander("Query and Results", expanded=False):
            st.subheader("query")
            st.code(response["query"], language='sql')
            st.subheader("Result")
            st.info(response["result"])
        