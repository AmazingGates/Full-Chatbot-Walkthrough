# The next model we will be practicing with is an end to end medical chat bot called Llama-2-7B-Chat-GGML

# We will start be using the quantized model. This is a form of the model that doesn't run with langchain,
#but is a stand alone.

# The exact version of the model we will be downloading is the llama-2-7b-chat-ggmlv3.q4_0.bin

# Let's start with an overview of the project we will be building.


#                               MEDICAL CHAT BOT

# The first thing we'll be doing is using our own component.

# This is going to be our data integration.

# Our data is going to be a [PDF Files]

# And the types of PDF Files we will be using are Medical Books.

# The book that the instructor is using is called The Gale ENCYCLOPEDIA of MEDICINE(second edititon).

# The data from this book is what we will feed to our LLM.

# After we are done with the data integration, the next thing we will be doing is data extraction.

# This will be the data extraction phase of our project.

# Once we have extracted our data, we will create chunks.

# It is importatnt to break our data up into text chunks because breaks our data up into managable portions
#so that it it is easier for our model to handle.

# Once we have our chunks, we will create our embeddings.

# These embeddings will be our vector.

# When we are finished with our embeddings/vector, we can build our semantic index.

# The next step will be to build our knowledge base. 

# For our Knowledge Base we will be using the Pinecone Vector Store

# This is what our project flow chart looks like so far and will be considered our backend component.


#                                   [PDF Files]
#                                        |
#                                [Data Extraction]
#                               /        |        \
#                  [Text Chunks]   [Text Chunks]   [Text Chunks]
#                        |               |               |
#                  [Embeddings]    [Embeddings]    [Embeddings]
#                              \         |        /
#                             [Build Semantic Index]
#                                        |
#                                [Knowledge Base] --- Pinecone


# So let's say we have a user.

# Our user we raise some query, or ask a question.

# The first thing we need to do is convert that question into a query embed.

# We will then send that query to our knowledge database.

# This is because our knowledge base has all of data in the form of vectors.

# The knowledge base will return a ranked result.

# This means that it will give us the closest vector with respect to the query we submitted, or question
#we asked.

# Now what we need to do is get our large language model.

# In this scenario we would be using the Llama2 model.

# With the help of the LLM we can filter out what our Actual Response to our query is.

# This filtered result, or "Actual Response" is what we will send back to our user.

# This is the complete flow of our medical chatbot.

# Writing down our flow charts or our "Model Architecture" makes it much easier to write to our code.

# This is because we can visualize and conceptualize the steps we need to take before we begin writing
#the actual code.

# This is what the flow chart would look like for our user end.

#                               [User]<-------------|
#                                  |                |
#                                  |                |
#                                  |                |
#                              [Question]           |
#    [Knowledge Base]              |                |
#       |  |                       |                |
#       |  |                       |                |
#       |  |<----------------[Query Embed]          |
#       |                                       [Llama2]
#       |--------->[Ranked Result]----------------->|



# Now we will be getting into the understanding of the technology we will be using to build our model.

# Techstack Used
#-----------------
# 1. Programming Language ---> Python
# 2. Langchain ---> Generative AI Framework
# 3. Project Frontend/Web app ---> Flask
# 4. Large Language Model ---> Meta Llama 2
# 5. Vector Data Base ---> Pine Cone

# This is the Techstack we will be using for this project.
