# Demo App for the course Building AI Applications with ChatGPT and OpenAI LLMs
# by Sumudu Tennakoon
# Last Update: 2022-07-09

import openai
from openai import OpenAI
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
    client = OpenAI(api_key=openai_api_key)
    
    MODEL = "gpt-3.5-turbo"
    INSTRUCTIONS = F"Translate this text in {from_language} into {to_language}."

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": INSTRUCTIONS},
            {"role": "user", "content": input_text},
        ],
        temperature=0,
        max_tokens=200
    )
    
    output_text = response.choices[0].message.content
    usage = response.usage.model_dump()
    return INSTRUCTIONS, response.model_dump(), output_text, usage


with st.form("my_form"):
    input_text = st.text_area("Enter text:", "")
    submitted = st.form_submit_button("Submit")
    if submitted:
        openai.api_key = openai_api_key

        instruction_prompt, response, output_text, usage  = translate(input_text, from_language, to_language)
        st.info(output_text)

        with st.expander("Prompt"):
            st.code(instruction_prompt)

        with st.expander("Response"):
            st.write(response)