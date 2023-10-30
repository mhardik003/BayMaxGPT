### BayMax : A Medical Chatbot
BayMax is a medical chatbot that can answer questions related to medicines (in this case, but can be used over any of the documents that you want to run it over) and it gives the answers by fetching the facts given in the data in realtime, thus making it a very fast and efficient chatbot.

* The model successfully runs on a laptop with 16GB of RAM and 6GB of VRAM

### Advantages of BayMax
* Run on baremetal (both on laptops and mobile phones)
* Doesn't need any sort of internet connection
* Doesn't hallucinate over random knowledge as it uses the facts provided in the data to generate the answer
* Data Agnostic : Can be used over any of the documents that you want to run it over as it uses RAG technique and doesn't need any sort of finetuning

#### How it works
* We first create the vector database using `MiniLM` embeddings and `Faiss` indexing
* Then we use `RAG` to fetch the answers from the database by converting the Queries given by the user into embeddings and then using `Faiss` to fetch the top 5 most similar embeddings and then using `RAG` to fetch the answers from the database
* We then send these answers to the model (here we have used quantized [Mistral 7B](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GPTQ))  along with the original query to generate the final answer
* In the webapp, if the user chooses the option to use the context from the data that has been provided, then we also show the context which the model has used to generate the answer to cross verify the legitimacy of the answer


### Future Works
* Adding more models
* Using the links in the mashqa dataset and provide them in the context as well to give one more step for the users to cross verify the answer and check the legitimacy of the answer given by the model
* Adding more data to the dataset to make the model more robust


### Instructions to run the model on a new dataset / Understand the working of the code

* Install the dependencies using `pip install -r requirements.txt`
* Unzip the dataset from [mashqa.zip](./Dataset/mashqa.zip) and place it in the `data` folder.
* Run [data_preprocess.ipynb](./data_preprocess.ipynb) notebook to clean the data and store it in the `cleaned_data` folder
* Run [RAG.ipynb](./RAG.ipynb) notebook to create the vector database and store it in the `vector_db` folder
### Built by
* [Hardik Mittal](https://github.com/mhardik003)
* [Astitva Srivastava](https://github.com/AstitvaSri)
* [Arjun Dosajh](https://github.com/arjundosajh)
* [Mihika Sanghi](https://github.com/mihikasanghi)

as part of [Megathon '23](https://megathon.in/)
