# Demo App for the course Building AI Applications with ChatGPT and OpenAI LLMs
# by Sumudu Tennakoon
# Last Update: 2022-07-09

import openai
import streamlit as st
import configparser
import os

config = configparser.ConfigParser()
config.read('config.ini')

openai_api_key = config['SECRETS']['openai_api_key']

st.title("🤖 Chatbot")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():

    openai.api_key = openai_api_key
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    messages = st.session_state.messages
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", 
                                            messages=messages,
                                            temperature=0,
                                            n=1,
                                            max_tokens=200,
                                            presence_penalty=0,
                                            frequency_penalty=0,
                                            )
    msg = response.choices[0].message
    prompt_tokens_count = response["usage"]["prompt_tokens"]

    st.session_state.messages.append(msg)
    st.chat_message("assistant").write(msg.content)

    st.info(F"Promot Tokens Count: {prompt_tokens_count}")
    with st.expander("Prompt"):
        st.write(messages)

    with st.expander("Response"):
        st.write(response)

