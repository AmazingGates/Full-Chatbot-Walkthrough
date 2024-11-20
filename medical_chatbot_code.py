from langchain import PromtTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.llms import ctransformers


PINECONE_API_KEY = "c1489634-2122-4dc3-a641-e16b3399b247"

PINECONE_API_ENV = "Serverless"

def load_pdf(Data):
    loader = DirectoryLoader(Data,
                    glob = "*.pdf",
                    load = PyPDFLoader)
    
    documents = loader.load()

    return documents

extracted_data = load_pdf("Data/")

extracted_data

def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap =20)
    text_chunks = text_splitter.split_documents(extracted_data)

    return text_chunks

text_chunks = text_split(extracted_data)

print("Length of my chunk:", len(text_chunks))

def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

    return embeddings

embeddings = download_hugging_face_embeddings()

embeddings

pinecone.init(api_key = PINECONE_API_KEY,
              environment = PINECONE_API_ENV)

index_name = "medical-chatbot"

docsearch = Pinecone.from_texts([t.page_content for t in text_chunks], embeddings, index_name = index_name)

docsearch = Pinecone.from_existing_index(index_name, embeddings)

query = "What are allergies?"

docs = docsearch.similarity_search(query, k = 3)

print("Result", docs)

prompt_template = """
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

PROMPT = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
chain_type_kwargs = {"prompt": PROMPT}

llm = ctransformers(model = "model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                    model_type = "llama",
                    config = {"max_new_tokens": 512,
                              "tempertaure": 0.8})

qa = RetrievalQA.from_chain_type(
    LLm = llm,
    chain_type = "stuff",
    retriever = docsearch.as_retriever(search_kwargs = {"k": 2}),
    return_source_documents = True,
    chain_type_kwargs = chain_type_kwargs)

while True:
    user_input = input(f"Input Prompt:")
    result = qa({"query": user_input})
    print("Response : ", result["result"])



llm = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Llama-2-7B-Chat-GGML", # Load the model from the correct repository
    model_file="llama-2-7b-chat.ggmlv3.q4_0.bin", # Specify the filename
    model_type="llama",
    max_new_tokens = 512, 
    temperature = 0.8 # Pass the config dictionary directly
)