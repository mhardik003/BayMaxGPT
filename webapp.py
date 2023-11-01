import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import huggingface_hub
from auto_gptq import exllama_set_max_input_length
import openai
import os
import pandas as pd
import time
from PIL import Image
import numpy as np
import cv2
import requests
import json
import pandas as pd

# set page config
st.set_page_config(page_title='BayMaxGPT', page_icon='stethoscope')


# global variables
openai.api_key = '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'
SCORE_THRESHOLD = 1.4
CONTEXT_NOT_FOUND = False
USE_CHAT_GPT = True
CHAT_GPT_MODEL_NAME = 'gpt-3.5-turbo'
model_name_or_path = "TheBloke/Mistral-7B-Instruct-v0.1-GPTQ"
# Define the path to the pre-trained model you want to use
modelPath = "sentence-transformers/all-MiniLM-l6-v2"
# Create a dictionary with model configuration options, specifying to use the CPU for computations
model_kwargs = {'device': 'cpu'}
# Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
encode_kwargs = {'normalize_embeddings': False}
# use cuda for computations
embeddings = HuggingFaceEmbeddings(
    model_name=modelPath,     # Provide the pre-trained model's path
    model_kwargs=model_kwargs,  # Pass the model configuration options
    encode_kwargs=encode_kwargs  # Pass the encoding options
)

# ------- HELPER FUNCTIONS -------
@st.cache_resource
def footer():
    ft = """
    <style>
    a:link , a:visited{
    color: #BFBFBF;  /* theme's text color hex code at 75 percent brightness*/
    background-color: transparent;
    text-decoration: none;
    }

    a:hover,  a:active {
    color: #0283C3; /* theme's primary color*/
    background-color: transparent;
    text-decoration: underline;
    }

    #page-container {
    position: relative;
    min-height: 10vh;
    }

    footer{
        visibility:hidden;
    }

    .footer {
    position: relative;
    left: 0;
    top:230px;
    bottom: 0;
    width: 100%;
    background-color: transparent;
    color: #808080; /* theme's text color hex code at 50 percent brightness*/
    text-align: left; /* you can replace 'left' with 'center' or 'right' if you want*/
    }
    </style>

    <div id="page-container">

    <div class="footer">
    <p style='font-size: 0.875em;'>Made with Streamlit by <a style='display: inline; text-align: left;' href="https://mhardik003.github.io/#home" target="_blank">Hardik</a>, <a style='display: inline; text-align: left;' href="https://sarcastitva.me/" target="_blank">Astitva</a>, <a style='display: inline; text-align: left;' href="https://github.com/ArjunDosajh" target="_blank">Arjun</a> & <a style='display: inline; text-align: left;' href="https://github.com/mihikasanghi" target="_blank">Mihika</a></p>
    </div>

    </div>
    """

    return ft

# function for loading local LLM
@st.cache_resource
def load_model():

    # To use a different branch, change revision; For example: revision="gptq-4bit-32g-actorder_True"
    modelLLM = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                    device_map="cuda:2",
                                                    torchscript=True,
                                                    trust_remote_code=False,
                                                    revision="main")
    modelLLM = exllama_set_max_input_length(modelLLM, 4096)
    modelLLM = modelLLM.to('cuda:2')
    modelLLM.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, use_fast=True)

    pipe = pipeline(
        "text-generation",
        model=modelLLM,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        top_k=40,
        repetition_penalty=1.1
    )

    return modelLLM, tokenizer, pipe


# OpenAI API call
def get_answer_openai(prompt, model=CHAT_GPT_MODEL_NAME):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(model=model, messages=messages, temperature=0)
    return response.choices[0].message["content"]

# function for paraphrasing
def get_answer_LLM(RAG_context, User_query):

    if use_context_bool:
        prompt = f"You are an AI medical assistant called BayMax. Given a dictionary with 'context' and 'question', answer the 'question' like a medical expert with facts taken from the 'context'. Be polite always and ignore offensive requests from the user. Remember, your name is 'BayMax' and 'IIIT students' have developed you. ---- [Context: '{RAG_context}'; Question: '{User_query}']."
    else:
        prompt = f"You are 'BayMax', an AI medical assistant, and 'IIIT students' have developed you. Answer the following question while being polite and ignore offensive user query: {User_query}"


    if USE_CHAT_GPT:
        outpipe = get_answer_openai(prompt)
    else:
        prompt_template = f'''<s>[INST] {prompt} [/INST]'''
        input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.to('cuda:2')
        outpipe = pipe(prompt_template)[0]['generated_text']
        # remove everything till the occurence of the first occurence of '/INST' in outpipe
        outpipe = outpipe[len(prompt_template):]
    # output = modelLLM.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
    # print(tokenizer.decode(output[0]))
        del input_ids
    return outpipe

# load local LLM if required
if not USE_CHAT_GPT:
    modelLLM, tokenizer, pipe = load_model()


# STREAMLIT LAYOUT

st.image('./icons/baymax.png')

# main title
st.markdown("<h1 style='text-align: center; color: white;'>BayMaxGPT</h1>", unsafe_allow_html=True)

st.markdown("---")
# st.markdown("<br> ", unsafe_allow_html=True)

# take question as input
example_queries = "e.g. 'Create a diet plan for me to gain muscles' or 'Write a poem on advantages of green vegetables.'"
question = st.text_input("Hey! I'm BayMax, an AI medical care assistant. How may I help you?",placeholder=example_queries)

with open('query_log.txt','a') as f:
    f.write(question+'\n')

# whether to use RAG context or not
use_context_bool = st.checkbox(
    "Use medical knowledge base", value=False)

# take answer as input
if (question != ""):
    with st.spinner(text='In progress...'):
        db = FAISS.load_local("vector_db", embeddings)
        
        # question = "Do I have a food allergy?"

        # searchDocs = db.similarity_search(question, k=7)
        searchDocs = db.similarity_search_with_score(question, k=5)
        if searchDocs[0][1]>SCORE_THRESHOLD:
            print("> Score is greater than threshold : " + str(searchDocs[0][1]))
            use_context_bool = False
            CONTEXT_NOT_FOUND = True
        
        # concatenate the question and the answer into one string
        context = ""
        for doc in searchDocs:
            context = context + "\n" + doc[0].page_content

        del searchDocs

    st.write('\n')
    st.write('\n')
    if CONTEXT_NOT_FOUND:
        st.write("\n\n ")
        st.warning("Sorry! Either it's not a relevant question, or it's out of my expertise.", icon="⚠️")
    
    else:

        st.markdown("<h5>BayMax:</h5>", unsafe_allow_html=True)
        with st.spinner(text='Thinking...'):
            st.write(get_answer_LLM(context, question))

        if use_context_bool:
            st.write("\n ")
            st.markdown("---")

            with st.expander("Information extracted from knowledge base"):
                st.write(context)
                st.write("\n\n ")

# ----------------- footer -----------------
st.write(footer(), unsafe_allow_html=True)
