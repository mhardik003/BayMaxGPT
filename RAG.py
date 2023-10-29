from langchain.document_loaders import HuggingFaceDatasetLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import AutoTokenizer, pipeline
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader, TextLoader


# Define the path to the pre-trained model you want to use
modelPath = "sentence-transformers/all-MiniLM-l6-v2"

# Create a dictionary with model configuration options, specifying to use the CPU for computations
model_kwargs = {'device':'cpu'}

# Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
encode_kwargs = {'normalize_embeddings': False}

# use cuda for computations
embeddings = HuggingFaceEmbeddings(
    model_name=modelPath,     # Provide the pre-trained model's path
    model_kwargs=model_kwargs, # Pass the model configuration options
    encode_kwargs=encode_kwargs # Pass the encoding options
)


db = FAISS.load_local("vector_db", embeddings)

question = "Do I have a food allergy?"

searchDocs = db.similarity_search(question, k=7)
# print(len(searchDocs))

# concatenate the question and the answer into one string
context=""
for doc in searchDocs:
    context= context + "\n" + doc.page_content

print(context)

