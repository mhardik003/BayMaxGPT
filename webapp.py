import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_name_or_path = "TheBloke/Mistral-7B-Instruct-v0.1-GPTQ"
# To use a different branch, change revision
# For example: revision="gptq-4bit-32g-actorder_True"
@staticmethod
def load_model():
    modelLLM = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                device_map="auto",
                                                torchscript=True,
                                                trust_remote_code=False,
                                                revision="main")

    modelLLM.cuda()
    modelLLM.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

    return modelLLM, tokenizer

modelLLM, tokenizer = load_model()


# def get_answer_LLM(RAG_context, User_query):
#     prompt = f"Given a dictionary with 'context' and 'question', answer the 'question' based on 'context' ---- [Context: '{RAG_context}'; Question: '{User_query}']"
#     prompt_template=f'''<s>[INST] {prompt} [/INST]'''
#     input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
#     output = modelLLM.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
#     # print(tokenizer.decode(output[0]))
#     return tokenizer.decode(output[0])

# # Inference can also be done using transformers' pipeline

# print("*** Pipeline:")
# pipe = pipeline(
#     "text-generation",
#     model=modelLLM,
#     tokenizer=tokenizer,
#     max_new_tokens=512,
#     do_sample=True,
#     temperature=0.7,
#     top_p=0.95,
#     top_k=40,
#     repetition_penalty=1.1
# )



# # Define the path to the pre-trained model you want to use
# modelPath = "sentence-transformers/all-MiniLM-l6-v2"

# # Create a dictionary with model configuration options, specifying to use the CPU for computations
# model_kwargs = {'device':'cpu'}

# # Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
# encode_kwargs = {'normalize_embeddings': False}

# # use cuda for computations
# embeddings = HuggingFaceEmbeddings(
#     model_name=modelPath,     # Provide the pre-trained model's path
#     model_kwargs=model_kwargs, # Pass the model configuration options
#     encode_kwargs=encode_kwargs # Pass the encoding options
# )




# # Title
st.title('MedQuery GPT')

# # take question as input
# question = st.text_input('Enter your question here:')

# # take answer as input
# st.header('Context found from the Dataset:')

# if(question != ""):
#     with st.spinner(text='In progress...'):
#         db = FAISS.load_local("vector_database_final", embeddings)

#         # question = "Do I have a food allergy?"

#         searchDocs = db.similarity_search(question, k=7)
#         # print(len(searchDocs))

#         # concatenate the question and the answer into one string
#         context=""
#         for doc in searchDocs:
#             context= context + "\n" + doc.page_content

#         st.write(context)



#     st.header('Answer from the model:')
#     st.write(get_answer_LLM(context, question))
