import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import huggingface_hub


st.set_page_config(page_title='MedQuery-GPT',
                   page_icon='./images/doctor.jpg')
# # Title
# st.title('MedQuery-GPT')

# set the title to the center of the page
st.markdown(
    "<h1 style='text-align: center; color: white;'>MedQuery-GPT</h1>", unsafe_allow_html=True)

st.markdown("---")
# st.markdown("<br> ", unsafe_allow_html=True)

# take question as input
question = st.text_input('Enter your query here')

model_name_or_path = "TheBloke/Mistral-7B-Instruct-v0.1-GPTQ"
# To use a different branch, change revision
# For example: revision="gptq-4bit-32g-actorder_True"


@st.cache_resource
def load_model():
    # huggingface_hub.login(token='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')

    modelLLM = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                    device_map="auto",
                                                    torchscript=True,
                                                    trust_remote_code=False,
                                                    revision="main")

    modelLLM.cuda()
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


modelLLM, tokenizer, pipe = load_model()

use_context_bool = st.checkbox(
    "Use context from provided dataset", value=False)

SCORE_THRESHOLD = 1.25
CONTEXT_NOT_FOUND = False

def get_answer_LLM(RAG_context, User_query):

    if use_context_bool:
        prompt = f"Given a dictionary with 'context' and 'question', answer the 'question' based on 'context'. ---- [Context: '{RAG_context}'; Question: '{User_query}'].'"
    else:
        prompt = f'Answer the following question: "{User_query}"'
    prompt_template = f'''<s>[INST] {prompt} [/INST]'''
    input_ids = tokenizer(
        prompt_template, return_tensors='pt').input_ids.cuda()
    outpipe = pipe(prompt_template)[0]['generated_text']
    # output = modelLLM.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
    # print(tokenizer.decode(output[0]))
    # remove everything till the occurence of the first occurence of '/INST' in outpipe
    outpipe = outpipe[len(prompt_template):]

    del input_ids

    return outpipe

# Inference can also be done using transformers' pipeline


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


# take answer as input
# st.header('Context found from the Dataset:')

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
        st.warning("Either the question doesn't seem like a medical query, or this knowledge is not there in my books.", icon="⚠️")
    
    else:

        st.markdown("<h5>Response</h5>", unsafe_allow_html=True)
        with st.spinner(text='Generating...'):
            st.write(get_answer_LLM(context, question))

        if use_context_bool:
            st.write("\n ")
            st.markdown("---")

            with st.expander("Context extracted from the dataset"):
                st.write(context)
                st.write("\n\n ")


