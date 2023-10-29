import openai
import os
import pandas as pd
import time


openai.api_key = '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'


def get_completion(prompt, model="gpt-3.5-turbo"):

    messages = [{"role": "user", "content": prompt}]

    response = openai.ChatCompletion.create(model=model, messages=messages, temperature=0)

    return response.choices[0].message["content"]


    

prompt = "Given the context in the text below, answer the question. Context : `` \n Question : `What are tips for managing my bipolar disorder?`"

response = get_completion(prompt)

print(response)