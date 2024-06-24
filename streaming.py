import base64
import os
import re

from openai import OpenAI
import PyPDF2
import streamlit as st
from PyPDF2 import PdfReader
import requests
from langchain import OpenAI
from langchain.adapters import openai
from langchain.chains import ConversationChain
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain.evaluation import load_evaluator, EvaluatorType
from langchain.memory import ConversationBufferMemory
import time

import tempfile


import openai
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_text_splitters import CharacterTextSplitter

# Load environment variables
load_dotenv()
def main():
 def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    num_pages = len(pdf_reader.pages)
    content = ""
    for page_num in range(num_pages):
        content += pdf_reader.pages[page_num].extract_text()
    return content


 os.environ["OPENAI_API_KEY"] == st.secrets["OPENAI_API_KEY"],

# Initialize the OpenAI model with streaming enabled
 llm = OpenAI(
    model_name='gpt-4o',
    temperature=0,
    max_tokens=100,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

# Initialize the conversation memory
 memory = ConversationBufferMemory()

# Streamlit app layout
 st.title('My Chatbot')

 option = st.selectbox('Choose an option:', ['Chat', 'Student Evaluation','Essay-Writing-handwritten'])
 if option=='Chat':
     with st.form(key='my_form'):
       user_input = st.text_input("Enter your input")
       submit_button = st.form_submit_button(label='Submit')
     chat = ChatOpenAI(temperature=0)
     full_response=""
     message_placeholder=st.empty()
     if "messages" not in st.session_state:
        st.session_state.messages = [SystemMessage(content="You are a helpful assistant.")]
     if submit_button:
      st.session_state.messages.append(HumanMessage(content=user_input))
     for part in chat.stream(st.session_state.messages):
       stream=part.content
       if(stream==" "):
         full_response+=" "
       full_response=full_response+stream
       time.sleep(0.05)
       message_placeholder.info(full_response)
       st.session_state.messages.append(AIMessage(content=part.content))

 elif option == 'Student Evaluation':
       subject = st.selectbox('Select Subject:', ['Essay-Writing'])
       uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
       if uploaded_file:
         doc_reader = PdfReader(uploaded_file)
         # read data from the file and put them into a variable called raw_text
         raw_text = ''
         for i, page in enumerate(doc_reader.pages):
           text = page.extract_text()
           if text:
               raw_text += text
       # Splitting up the text into smaller chunks for indexing
         text_splitter = CharacterTextSplitter(
           separator="\n",
           chunk_size=1000,
           chunk_overlap=200,  # striding over the text
           length_function=len,
         )
         texts = text_splitter.split_text(raw_text)
         full_stream="\nTHESIS CLARITY :  \n"
         embeddings = OpenAIEmbeddings()
         docsearch = FAISS.from_texts(texts, embeddings)
         chain = load_qa_chain(OpenAI(),
                             chain_type="stuff")
         query="evaluate the given message, Evaluation criteria: 1. Thesis clarity: Clear and focused thesis statement that addresses the essay prompt 2. Analysis Depth: In-depth analysis of the topic with supporting evidence 3. Organization: Well structured and logically organized essay 4. Writing Clarity: Clear and concise writing with proper grammar and punctuation 5. Conclusion"
         message=st.empty()
         if query:
           docs = docsearch.similarity_search(query)
           stream=chain.run(input_documents=docs, question=' write a Clear and focused thesis statement that addresses the essay ')
           for word in stream:
             full_stream=full_stream+word
             time.sleep(0.005)
             message.info(full_stream)

           full_sentence="\n ANALYSIS DEPTH : \n "
           msg=st.empty()
           sentence=chain.run(input_documents=docs, question='write a In-depth analysis of the topic with supporting evidence')
           for word in sentence:
             full_sentence = full_sentence + word
             time.sleep(0.005)
             msg.info(full_sentence)


           mess=st.empty()
           full="\n  ORGANIZED EASSY : \n "
           s=chain.run(input_documents=docs, question=' write a Well structured and logically organized essay based on the topic of the pdf')
           for word in s:
             full = full + word
             time.sleep(0.005)
             mess.info(full)



           m=st.empty()
           f="\nWRITING ClARITY :\n"
           write=chain.run(input_documents=docs, question='see if the text contains any'
                                                         ' gramatical errors or not')
           for word in write:
             f = f + word
             time.sleep(0.005)
             m.info(f)

           messag=st.empty()
           full_strea='\nCONCLUTION :\n '
           strea=chain.run(input_documents=docs, question='write down a conclution')
           for word in strea:
             full_strea = full_strea + word
             time.sleep(0.005)
             messag.info(full_strea)

 elif option=='Essay-Writing-handwritten':
       uploaded_file = st.file_uploader("Choose a handwritten file", type="jpg")
       ##uploaded_file=str(uploaded_file)
       ##temp_path =  re.search(r"/(\w+)/\w+-\w+-\w+-\w+-\w+", uploaded_file,re.IGNORECASE)
       ##file_id = temp_path.group(0)
       ##st.write(file_id)

       import base64
       import requests

       # OpenAI API Key
       os.environ["OPENAI_API_KEY"] == st.secrets["OPENAI_API_KEY"],
       # Function to encode the image

       def encode_image(image_file):
           return base64.b64encode(image_file.read()).decode('utf-8')

       # Path to your image
       image_path = uploaded_file

       # Getting the base64 string
       base64_image = encode_image(image_path)

       headers = {
           "Content-Type": "application/json",
           "Authorization" : f"Bearer {st.secrets['API_KEY']}"
       }

       payload = {
           "model": "gpt-4o",
           "messages": [
               {
                   "role": "user",
                   "content": [
                       {
                           "type": "text",
                           "text": "check for any grammatical errors in the image"
                       },
                       {
                           "type": "image_url",
                           "image_url": {
                               "url": f"data:image/jpeg;base64,{base64_image}"
                           }
                       }
                   ]
               }
           ],
           "max_tokens": 300
       }

       response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

       ans=response.content
       import json

       # The provided JSON response
       response_json = ans  # Replace with the actual JSON data

       # Parse the JSON
       response_dict = json.loads(response_json)

       # Extract the content from the assistant's message
       content = response_dict["choices"][0]["message"]["content"]

       # Remove escape characters (e.g., '\\n' becomes '\n')
       content_cleaned = content.replace("\\n", "\n")

       answer=content_cleaned
       messages = st.empty()
       full_answer=""
       for word in answer:
           full_answer=full_answer+word
           time.sleep(0.005)
           messages.info(full_answer)

import time
import sys



# Run the main function
if __name__ == '__main__':
 main()

