# Demo App for the course Building AI Applications with ChatGPT and OpenAI LLMs
# by Sumudu Tennakoon
# First Upload: 2023-07-09
# Last Update: 2023-12-17
# Reference: https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps

import openai
from openai import OpenAI
import streamlit as st
import configparser
import os

config = configparser.ConfigParser()
config.read('config.ini')

openai_api_key = config['SECRETS']['openai_api_key']

client = OpenAI(api_key=openai_api_key)

CHATBOT_NAME = 'Chatty'
MODEL = "gpt-3.5-turbo"
MAX_TOKENS = 200
INSTRUCTIONS = F"You are a helpful chatbot assitant named {CHATBOT_NAME}. Your goal is to assist users with accurate and respectful responses."

def initialize_chat_messages():
    return [{"role": "system", "content": INSTRUCTIONS},
            {"role": "assistant", "content": F"Hi I am your AI assitant {CHATBOT_NAME}. How can I help you?"}]

st.title("🤖 Chatbot")

# Intilize Chat Thread
if "messages" not in st.session_state:
    st.session_state["messages"] = initialize_chat_messages()

with st.sidebar:
    st.session_state["max_questions"] = st.number_input("Max Questions:", min_value=1, max_value=10, step=1, value=5, format='%d')

for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

if prompt := st.chat_input("Ask your question here..."):

    # Add user question to chat thread
    st.session_state["messages"].append({"role": "user", "content": prompt})

    # Display user question
    with st.chat_message("user"):
        st.markdown(prompt)

    messages=[{"role": msg["role"], "content": msg["content"]} for msg in st.session_state.messages]

    response = client.chat.completions.create(model=MODEL, 
                                            messages=messages,
                                            temperature=0,
                                            n=1,
                                            max_tokens=MAX_TOKENS,
                                            presence_penalty=0,
                                            frequency_penalty=0,
                                            )
    msg = response.choices[0].message.model_dump()
    prompt_tokens_count = usage = response.usage.prompt_tokens

    st.session_state.messages.append(msg)
    st.chat_message("assistant").write(msg["content"])

    st.info(F"Prompt Tokens Count: {prompt_tokens_count}")
    with st.expander("Prompt Messages"):
        st.write(messages)

    with st.expander("Response"):
        st.write(response.model_dump())

    if len(st.session_state["messages"])-2 >= st.session_state["max_questions"] *2: # -2 to exclude system prompt and intial assitant prompt
        st.error(F"Exceed {st.session_state['max_questions'] } questions. Resetting the chat history. New questions will not use the past chat history.")
        st.session_state["messages"] = initialize_chat_messages()