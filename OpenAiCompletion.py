from tkinter import filedialog

import PyPDF2
import streamlit as st
from PyPDF2 import PdfReader

from langchain.adapters import openai

from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
import time

import tempfile


import openai
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms.openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import CharacterTextSplitter

# Load environment variables
load_dotenv()

def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    num_pages = len(pdf_reader.pages)
    content = ""
    for page_num in range(num_pages):
        content += pdf_reader.pages[page_num].extract_text()
    return content

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

# Define the main function
def main():
    # Initialize the chat model with the memory and streaming enabled
   option = st.selectbox('Choose an option:', ['Chat', 'Student Evaluation'])
   if option=='Chat':
     with st.form(key='my_form'):
           user_input = st.text_input("Enter your input")
           submit_button = st.form_submit_button(label='Submit')
     chat = ChatOpenAI(temperature=0)
     full_response=""
     message=""
     message_placeholder=st.empty()

     # Check if 'messages' exists in session state, otherwise initialize it
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
       if(uploaded_file):
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

         embeddings = OpenAIEmbeddings()
         docsearch = FAISS.from_texts(texts, embeddings)
         chain = load_qa_chain(OpenAI(),
                             chain_type="stuff")
         query = st.text_input("Enter what you wanna ask")
         if(query):
           docs = docsearch.similarity_search(query)
           chain = chain.run(input_documents=docs, question=query)

         custom_criteria = ("system",
                            "Custom criteria: 1. Creativity: Original and innovative ideas 2. Coherence: Logical flow of arguments 3. Evidence: Strong supporting evidence 4. Engagement: Captivating writing style")

         prompt = ChatPromptTemplate.from_messages([
             ("system",
              "evaluate the given message, Evaluation criteria: 1. Thesis clarity: Clear and focused thesis statement that addresses the essay prompt 2. Analysis Depth: In-depth analysis of the topic with supporting evidence 3. Organization: Well structured and logically organized essay 4. Writing Clarity: Clear and concise writing with proper grammar and punctuation 5. Conclusion"),
             ("user", f"{docsearch}"),
             custom_criteria  # Add the custom criteria here
         ])

         # Use the prompt in your OpenAI chat model
         st.write(prompt)



import time




# Run the main function
if __name__ == '__main__':
    main()

