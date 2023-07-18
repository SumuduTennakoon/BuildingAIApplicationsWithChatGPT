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

st.title("📑 Translator")

#from_language = "English"
#to_language = "Sinhala"

from_language = st.selectbox(
    "From Language:",
    ("English", "French", "Japanese", "Sinhala", "Spanish", "Tamil"),
    label_visibility="visible",
    disabled=False,
    index=0,
    key="from_language"
)

to_language = st.selectbox(
    "To Language:",
    ("English", "French", "Japanese", "Sinhala", "Spanish", "Tamil"),
    label_visibility="visible",
    index=3,
    disabled=False,
    key="to_language"
)

st.subheader(F"{from_language} -> {to_language}")

def translate(input_text, from_language, to_language):
    prompt = F"Translate this text in {from_language} into {to_language}:\n\n{input_text}\n\n"
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt= prompt,
        temperature=0.3,
        max_tokens=100,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    return prompt, response 


with st.form("my_form"):
    input_text = st.text_area("Enter text:", "")
    submitted = st.form_submit_button("Submit")
    if submitted:
        openai.api_key = openai_api_key

        prompt, response  = translate(input_text, from_language, to_language)
        st.info(response["choices"][0]["text"])

        with st.expander("Prompt"):
            st.code(prompt)

        with st.expander("Response"):
            st.write(response)