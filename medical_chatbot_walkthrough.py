# This is where we will start writing the code for our chat bot.

# The first thing we will do is get our requirements.

# The first requirement we will need is something called the ctrancformers.

# This is the specific version we will be using.

ctransformers==0.2.5

# This is the quantized model.

# The second library we will need is the sentence transformer.

# This is the version we need.

sentence-transformers==2.2.2

# The next thing is the pinecone vector data base.

# This is the version we will be using

pinecone-client

# And this is the version of langchain we will be using

langchain==0.00.225

# We will also need flask to create our frontend

flask

# Now in our main file (medical_chatbot_code) we will import the libraries we need.

from langchain import PromtTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.llms import ctransformers


# Next we will create a folder called data where we will put our data (Medical PDF)

# Now we will get our pinecone api key and implement it into our code.

PINECONE_API_KEY = "c1489634-2122-4dc3-a641-e16b3399b247"

# Now we need to create a Pinecone Enviornment by creating a API index on the pinecone website.

PINECONE_API_ENV = "Serverless"

# The next thing we will do is load our data from our PDF.

# We will do this by defining a function to do this for us.

def load_pdf(Data):
    loader = DirectoryLoader(Data,
                    glob = "*.pdf",
                    load = PyPDFLoader)
    
# Once we load our pdf we will need to call the load function. We will be storing the call function in 
#a variable.

    documents = loader.load()

# Then we will return our variable documents where our loader is stored.

    return documents

# Now we will extract our data with a function that we will store in a variable.

extracted_data = load_pdf("Data/")

# To double check that our data was extracted successfully we can run our variable with the load function
#stored in it.

# Note: This is only confirmation and isn't a necessity so we can comment this piece of code out once
#we are done with it.

extracted_data

# Now that we have how extraxted data, the next thing we will do is create our text chunks.

# Inside our defined text split we will be using the RecursiveCharacterTextSplitter which will take two
#parameters, our chunk size and our chunk over lap.

# We will also store our RecursiveCharacterTextSplitter in a variable called text splitter, for easier access.

# Once we have our text splitter we can split our data using a split documents function that will take one
#parameter.

# We will also store this split documents function in a variable called text chunks.

# Then we will return our text chunks.

def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap =20)
    text_chunks = text_splitter.split_documents(extracted_data)

    return text_chunks

# Now we can call our text split function and pass it a parameter of extracted data.

# We will also store our text split function in a text chunks variable.

text_chunks = text_split(extracted_data)

# we can also verify the length of our chunks by using a print method.

# Note: This isn't a necessary part of the code and we can comment it out once we verify.

print("Length of my chunk:", len(text_chunks))

# Next we will create another function to get vector embeddings.

# This function will download the embeddings package itself and will get stored in a variable called
#embeddings.

# Inside the huggingfaceembeddings function we need to pass as a parameter which is the actual model we 
#will be downloading.

# Then we need to return embeddings

def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

    return embeddings

# Now we will download the model by calling HuggingFaceEmbeddings which is stored in embeddings.

embeddings = download_hugging_face_embeddings()

# We can also check the specs of our model by calling embeddings.

# Note: This is not a necessary step and we can comment this out once we are done.

embeddings

# With the help of this embedding model we will be able to convert our text into embeddings/vectors.

# Next we will write the code that will initialize our Pinecone client

pinecone.init(api_key = PINECONE_API_KEY,
              enviornment = PINECONE_API_ENV)

# Next we will initialize our pinecone index name.

index_name = "medical-chatbot"

# Once that is done we'll call our pinecone to get our extracted text by poviding our embeddings model
#and our index name.

docsearch = Pinecone.from_texts([t.page_content for t in text_chunks], embeddings, index_name = index_name)

# What this code will do is take all of our chunks, as well as our embedding model, and our index name together
#and apply the embedding model to the data we have, which will convert our data to embeddings, then it will 
#automatically store that vector to the Pinecone Index we created on the Pinecone website.

# At this point we should be able to perform a semantic search.

docsearch = Pinecone.from_existing_index(index_name, embeddings)

# Here we are inquiring about a query.

query = "What are allergies?"

# Here we are searching our knowledge base, and we are requesting the top 3 similar results.

docs = docsearch.similarity_search(query, k = 3)

# Next we will print those results.

print("Result", docs)

# The ranked results we get back will not be readable at this point.

# This is where our LLM comes into play.

# The LLM will generate our "Actual Answer", and return it to us.

# These are the steps we need to take to start that process.

# First we'll have our prompt templates.

prompt_template = """
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

# Now that we have our prompt, we can create our prompt template.

PROMPT = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
chain_type_kwargs = {"prompt": PROMPT}

# Next we will load our Llama model.

llm = ctransformers(model = "model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                    model_type = "llama",
                    config = {"max_new_tokens": 512,
                              "tempertaure": 0.8})

# Once our model is loaded, the next thing we will do is create our question and answering object.

# For this we will use a Retrieval.

qa = RetrievalQA.from_chain_type(
    LLm = llm,
    chain_type = "stuff",
    retriever = docsearch.as_retriever(search_kwargs = {"k": 2}),
    return_source_documents = True,
    chain_type_kwargs = chain_type_kwargs)

# The next thing we will do is ask a question using a while True loop.

# This while True loop will take input from the user. Then it will ask the question to our llm.
#And finally our llm will print a response.

while True:
    user_input = input(f"Input Prompt:")
    result = qa({"query": user_input})
    print("Response : ", result["result"])

# Now we will ask some questions.

# The backend of our Medical Chatbot is complete.

# The next phase will be the frontend.


# Now we will begin the second half of our project.

# We will start by adding a new file called template.

# This template file will hold all of our logic.

# Next will import some of the libraries we will be using.

import os
from pathlib import Path
import logging

# The logging is import because it will create a record whenever we execute the template or any
#py file.

# The logging will show us the time and the date and whether or not the folder has been created.

# It will also show the location as well.

# Logging is a built in library of python and we can read the documentation to get a better understanding
#of it when we need to.

# This is the logging statement we will be using.

logging.basicConfig(level = logging.INFO, format = "[%(asctime)s]: %(message)s")

# Notice we have a logging level, which is basically saving the information we need, like why is
#the folder being created, what is the path, what time was it created etc.

# This is the information we will be logging.

# We also need to mention the format of the logging.

# This will include the structure of the time stamp, and what particular message will be displayed
#during the logging.

# To set the time we will be using the asci time stamp. 

# We will using this along side the message we want displayed during execuetion.

# Next we will create a list of the files and save it to a variable called list of files.

# The first parameter of our list will be the source folder, which we will call "src".

# Inside the "src" we will create a constructor method ( __init__ ).

# Our next list item will be helper.py file in the same folder.

# We will add one more file to the same folder and name it prompt.

# We will also need a .env file.

# Next will be the setup.py file.

# We will also be adding a research/trials ipynb file to keep data about our project trials.
#Note: we only used a ipynb file because the instructor has his model saved there. That is not where
#our model is located. We used Google Colab to create our Fully Functional Model.

# We also need an app.py

# Next we need a store index file.

# Now we will have a static file and a templates file.

# Inside the templates file we will create another file called chat html.

list_of_files = [
    "src/__init__.py",
    "src/helper.py",
    "src/prompt.py",
    ".env",
    "setup.py",
    "research/trials.ipynb",
    "app.py",
    "store_index.py",
    "static",
    "templates/chat.html"
]

# These are the folders and files we need as of now.

# Now that we have our files and folders we will go over the steps of implementing them.

# We will start by writing some of the logic.

# We will be using simple Python to write this logic.

# First we will loop through our list.

# Then we will convert our filepath to path.

# Then we will store that Path in a variable called filepath

for filepath in list_of_files:
    filepath = Path(filepath)


# The next thing we want to do is separate our py folders and files.

# First we will store our file directory followed by mour file name.

# Then we will use a os.path.split method with the parameter filepath.

# This method will split our folders from our files.

# Because we have two variables the first variable will represent our file directory, and the
#second variable will represent our file name.

    filedir, filename = os.path.split(filepath)

    # Now we will create our file directory. Here we are saying if the file directory doesn't
    #equal an empty space to create it.
    # We will then pass in the directory name of the directory we want to create. We will also pass
    #in the parameter exists_ok and set it to True.
    if filedir !="":
        os.makedirs(filedir, exists_ok = True)

        # Once this is done we want to log our information.
        logging.info(f"Creating Directory; {filedir} for the file {filename}")

    # Once our folders are created, we will create the files inside the folder
    # And if the filepath doesn't exist, we want to create it.
    # We will aslo check the size of the file.
    # Inside the getsize method we will pass the file name as a parameter.
    # We're also saying that if that file is not empty, we want to create it
    # We can do that with a with open method that takes as parameters the filepath, and w for write.
    # Then we will use a pass opertion to close our cose because we aren't writing anything else here.
    if (not os.path.exists(filepath)) or (os.path.getsize(filename) ==0):
        with open(filepath, "w") as f:
            pass
        

        # Now we also need to log this information.
        logging.info(f"Creating file: {filepath}")

        # And if it doesn't exist we want to create it.

    else:
        logging.info(f"{filename} is already created")


# This is our simple logic we have created.

# We can actually run the code now as is to create all of the files we have in our list so far.

# Now that we can see that it is working properly and creating our files, we will add a few more 
#files to our list to be created.

list_of_files = [
    "src/__init__.py",
    "src/helper.py",
    "src/prompt.py",
    ".env",
    "setup.py",
    "research/trials.ipynb",
    "app.py",
    "store_index.py",
    "static/.gitkeep",
    "templates/chat.html",
    "test.py"
]

# Note: We must compile the code everytime we make a change something in order to update the code.

# As we can see we can create as many files and folders as we want.

# So iin the future when we're working on large projects, instead of creating the folders and files one
#by one, we can use this simple template to create all of the files and folders we need at once.

# We can combine this code with the logic that goes along with it and it can save us a lot of time with formating.

# Next we will use the steup.py file to help us turn our src folder into a local package.

# This is will let our src folder, (or any folder we choose to use), behave just like any other package
#we install using pip and import into our code.

# Now we will begin writing the code for our setup.py file

# We will start by using a pre built python library called setuptools.

# From setuptools we will import find packages and setup.

from setuptools import find_packages, setup

# Next we will create our object and call it setup.

# This will take as parameters name, version, author, author email, packages, install requires.

# We also have a find_packages function.

# When called, this function will look for the constructor method (__init__.) in every folder.

# The folder we find the file in will be considered our local package.

# That's why we made put the constructor method inside the src folder. Because we planned on making that
#folder our local package.

setup(
    name = "medical Chatbot Project",
    version = "0.0.0",
    author = "Python Papi",
    author_email = "alphacomm7@gmail.com",
    packages = find_packages(),
    install_requires = []
)

# Now we can install the setup.py file.

# This is how we can do that.

# First we will add an item to our requirements.txt file. This is what we'll be adding ( -e . )

# What the ( -e . ) will do is once the requirements.txt file is installed, it will look for the
#setup.py fill automatically.

# And once it finds the setup.py file it will trigger the file to run and follow the instructions
#in the setup.py file.

# With that done and everything set up, we can now move on to working on our enviornment.

# The first thing we will be adding to this .env file is a copy of our pinecone key.

# The second thing we will add is the pinecone env.

# Note: If we are using a severless env on the pinecone website we don't need to specify a env
#and we can leave this part of the code out.

# We will be putting the api key and env inside this folder, as well as any other secret information
#we have that we don't want to share with the public.

# When transfer our code to github the env file doesn't get transfered.

# This is because of the gitignore feature of github.

# This feature treats env and a number of different file types as classified and won't transfer them
#into github when transfering your code over.

# To read this file we need a particular library. python-dotenv.

# We will add this library to our requirements.txt list

# Now that we have our secret file and we have the library we need to read it, we can start
#adding the componenets.

# We will start by adding the def load function to the helper.py file in our src folder.

# We also have to remember to bring over the imports that are needed for this load function.

# This is the import that we need (from langchain.document_loaders import PyPDFLoader, DirectoryLoader).

# The next thing we add to the helper.py file is the def text_splitter function.

# We must also bring over the imports for this too.

# This is the import we will be using (from langchain.text_splitter import RecursiveCharacterTextSplitter).

# Next we will add the def download_huggingface function.

# This is the import we will be using (from langchain.embeddings import HuggingFaceEmbeddings).

# Basically this helper file will hold our functions, which allows us to call them one by one when
#needed instead of having to reapeatedly write them out.

# Next we will be using the store.py file to push our vector to the vector db.

# If we want to push our vector to the vector db the first thing we need to done is load our
#pdf file which we are getting our data from.

# Next we will load the text_split function, and lastly the download_hugging_face_embeddings function.

# All of these import will come directly from our src folder which we designated to be our local package.

# Next we need to import the pinecone to this folder.

# These are the imports we will use to do that. (from langchain.vectorstores import Pinecone 
#                                                import pinecone)

# Then we need to load our load_pdf package.

# This is the import we will use to do that.

# from dotenv import load_dotenv

# This particular package will allows to read our .env file, where we keep sensitive and private information.

# We also need the os import.

# The next thing we want to do is load our dotenv file inside our store index file. 

# This is how we will start that process.

load_dotenv()

# Next we will use the os to read our private key, like this.

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

# Note: We will do the same thing for the environmemt, but only if we're using the pinecone enviornment.

PINECONE_API_ENV = os.environ.get("PINECONE_API_ENV")

# We can use the print function to check if our key and environment are being read properly.

print(PINECONE_API_KEY)
print(PINECONE_API_ENV)

# This is only optional and can be commented out or removed all together.

# Now we wre ready to load our pdf.

# This is how we will do that.

extracted_data = load_pdf("Data/")

# This line of code will load our pdf data.

# Next we need to implement a text splitter, to get our text chunks.

# This is the line of code we will use.

text_chunks = text_split(extracted_data)

# After we get our chunks we need to get our emeddings.

# This is the line of code we will use for that.

embeddings = download_hugging_face_embeddings()

# Next we want to initialize our pinecone.

# This is the code we will use to do that.

pinecone.init(api_key = PINECONE_API_KEY,
              environment = PINECONE_API_ENV)

# Next we want to store our data, and mention our pinecone index name before that.

# This is the code we will use to do that.

index_name = "medical-chatbot"

docsearch = Pinecone.from_texts([t.page_content for t in text_chunks], embeddings, index_name = index_name)

# We should be able to execute this file now and have our data stored in our pinecone db.

# With that, our store.py file is done.

# The next thing we need to do is add our prompt to our prompt.py file.

# This is the code we will be adding to our file.

prompt_template = """
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

# Now that this is done we will start working on our app.py file.

# The first thing we need to do is import flask into this file.

# We also need to import render_Template, jsonify and request.

# This is how we will do that.

from flask import Flask, render_Template, jsonify, request

# Inside this app.py file we also need to upload our embeddings.

# This is how we will do that.

from src.helper import download_hugging_face_embeddings

# We also need to intialize our pinecone inside this file.

# This is how we will do that.

# To create our question and answer feature to chat with our llm we will also need these imports in the
#app.py file.

from langchain.prompts import PromptTemplate
from langchain.llms import ctransformers
from langchain.chains import RetrievalQA

# And because we want to access the secrets we have i our .env file, we add this next import also.

from dotenv import load_dotenv

# We will also need to load our prompt template so we will use this import also.

from src.prompt import *

# We will also be importing the operating system.

import os

# With all of our imports done, the very first thing we will do is initialize our flask.

# This is how we will do that.

app = Flask(__name__)

# This is how we define our flask.

# The next thing we need to do is load our api environment.

# This is the code we will use to do that.

load_dotenv() 

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_API_ENV = os.environ.get("PINECONE_API_ENV")

# Next we will load our embedding mode.

# This is the code we will use to do that.

embeddings = download_hugging_face_embeddings()

# Next we need to initialize our pinecone, and our index name. 

# This is the code we will use to do that.

pinecone.init(api_key = PINECONE_API_KEY,
              environment = PINECONE_API_ENV)

index_name = "medical-chatbot"

# If we have an existing pinecone vector data base and we already have an idex name, we can use
#this code below.

docsearch = Pinecone.from_existing_index(index_name, embeddings)

# The next things we wil be adding to our app.py file are the prompt template, the chain type kwargs, and
#the llm equals ctransformers.

# These are the codes we will be using.

PROMPT = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])

chain_type_kwargs = {"prompt": PROMPT}

llm = ctransformers(model = "model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                    model_type = "llama",
                    config = {"max_new_tokens": 512,
                              "tempertaure": 0.8})

# The next thing we will do is initialize our QA object.

# This is the code we will use to do that.

qa = RetrievalQA.from_chain_type(
    LLm = llm,
    chain_type = "stuff",
    retriever = docsearch.as_retriever(search_kwargs = {"k": 2}),
    return_source_documents = True,
    chain_type_kwargs = chain_type_kwargs)

# Next we will create the default route of our flask.

# This is the code we will use to do that.

@app.route("/") # This is our decorator
def index():
    return render_template("chat.html")

# This is how we will execute this code.

if __name__ == "__main__":
    app.run(debug = True)

# Next we will be working in the chat.html file.

# The first thing we will do is find some example html code documentation that we can copy and
#paste, (unless we want to write our own) to use as our test code.

# This is the code we will be using.

/* Set height to 100% for body and html to enable the background image to cover the whole page: */
body, html {
  height: 100%
}

.bgimg {
  /* Background image */
  background-image: url('/w3images/forestbridge.jpg');
  /* Full-screen */
  height: 100%;
  /* Center the background image */
  background-position: center;
  /* Scale and zoom in the image */
  background-size: cover;
  /* Add position: relative to enable absolutely positioned elements inside the image (place text) */
  position: relative;
  /* Add a white text color to all elements inside the .bgimg container */
  color: white;
  /* Add a font */
  font-family: "Courier New", Courier, monospace;
  /* Set the font-size to 25 pixels */
  font-size: 25px;
}

/* Position text in the top-left corner */
.topleft {
  position: absolute;
  top: 0;
  left: 16px;
}

/* Position text in the bottom-left corner */
.bottomleft {
  position: absolute;
  bottom: 0;
  left: 16px;
}

/* Position text in the middle */
.middle {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  text-align: center;
}

/* Style the <hr> element */
hr {
  margin: auto;
  width: 40%;
}

# Now we can run our app.py

# When the code is configured the way it should be, we will see a portal number for our local chat, and 
#we'll be able to view our html page.

# We'll get a host and a port number, but there is a way specify which port number we want to use.

# We can do that with this code below.

if __name__ == "__main__":
    app.run(host = "0.0.0.0", port = 8080, debug = True)

# When everything is configured properly, we will have just specified to our code to host on port
#8080


# Next we can visit the Bootstrap website.

# This is a website where we can get templates to use in our test code.

# Next we can simply search the web for chatbot, html and css templates and use one that we can like.

# The results should return us a bunch of different websites about chatbot, html and css templates.

# We can simply choose the ones we like, download it, and play around with them to get more familiar with 
#them and learn how to modify things to our specification.

# We are going to use a tst html code that we copied from the internet.

# This is the test html code we will be using to do that.

<div class = "bgimg">
    <div class = "topleft">
      <p>Logo<p>
    </div>
    <div class = "middle">
      <h1>COMING SOON</h1>
      <hr>
      <p>35 Days</p>
    </div>
    <div class = "bottomleft">
      <p>Some Text</p>
    </div>
</div>


# Once we have picked the html and the css we want to use we will move forward.

# The next thing we will be doing is adding our finally route.

# This is how we will do that.

@app.route("/get", methods = ["GET",  "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result = qa({"query": input})
    print("Response : ", result["result"])
    return str(result["result"]) 

# This finally app will be going into our app.py file with the other route definitions.

# What we're doing inside our final route is taking a passage entered by the user and storing
#it in a variable called msg. Next we will store our msg variable inside a variable called input.
#From here we can print out our input.

# After that we will be sending the query to our QA, because our QA is already defined.

# After that we will get the response and print it out.

# Then we are also sending that response to our U I.

# At this point we would execute our app.py file to make everything is working the it's suppose to.

