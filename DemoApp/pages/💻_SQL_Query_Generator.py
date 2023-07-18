# Demo App: Building AI Applications with ChatGPT and OpenAI LLMs
# by Sumudu Tennakoon
# Last Update: 2022-07-09

import openai
import streamlit as st
import configparser
import os

config = configparser.ConfigParser()
config.read('config.ini')

openai_api_key = config['SECRETS']['openai_api_key']

st.title("↔💻 SQL Query Generator")

def query_generate(db_schema, user_prompt):
    db_schema = db_schema.replace("\n", "\n# ")
    prompt = F"### SQL tables, with their properties:\n# \n# {db_schema}\n#\n### {user_prompt}"
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0,
        max_tokens=150,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["#", ";"]
    )
    return prompt, response

with st.form("my_form"):
    db_schema = st.text_area("DB Schema:", "")
    user_prompt = st.text_area("User Prompt:", "")
    submitted = st.form_submit_button("Submit")
    if submitted:
        openai.api_key = openai_api_key

        prompt, response = query_generate(db_schema, user_prompt)
        st.code(response["choices"][0]["text"], language='sql')

        with st.expander("Prompt"):
            st.code(prompt)

        with st.expander("Response"):
            st.write(response)