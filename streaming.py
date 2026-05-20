import base64
import os
import requests
import json
import time
import openai
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv

# LangChain / community adapters (used selectively below)
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()


def main():
  def read_pdf(file):
    pdf_reader = PdfReader(file)
    num_pages = len(pdf_reader.pages)
    content = ""
    for page_num in range(num_pages):
      text = pdf_reader.pages[page_num].extract_text()
      if text:
        content += text
    return content

  # Derive API key from environment or Streamlit secrets (safe when secrets.toml is missing)
  api_key = os.environ.get("OPENAI_API_KEY", "")
  try:
    if hasattr(st, "secrets"):
      secret_val = st.secrets.get("OPENAI_API_KEY")
      if secret_val:
        api_key = secret_val
  except FileNotFoundError:
    # No Streamlit secrets file; keep using environment variable
    pass

  # Configure openai python client for fallback usage
  if api_key:
    openai.api_key = api_key

  # Initialize the conversation memory
  memory = ConversationBufferMemory()

  # Streamlit app layout
  st.title('My Chatbot')

  option = st.selectbox('Choose an option:', ['Chat', 'Student Evaluation', 'Essay-Writing-handwritten'])

  if option == 'Chat':
    with st.form(key='my_form'):
      user_input = st.text_input("Enter your input")
      submit_button = st.form_submit_button(label='Submit')

    use_openai_fallback = False
    try:
      chat = ChatOpenAI(temperature=0, openai_api_key=api_key)
    except Exception as e:
      # Fall back to the openai python client if ChatOpenAI initialization fails
      st.warning(f"ChatOpenAI init failed; falling back to OpenAI client: {e}")
      use_openai_fallback = True
    full_response = ""
    message_placeholder = st.empty()
    if "messages" not in st.session_state:
      st.session_state.messages = [SystemMessage(content="You are a helpful assistant.")]
    if submit_button and user_input:
      st.session_state.messages.append(HumanMessage(content=user_input))
    # Stream responses if the model supports it
    if use_openai_fallback:
      # Build messages list for the OpenAI API
      api_messages = []
      for m in st.session_state.messages:
        role = "system" if isinstance(m, SystemMessage) else ("user" if isinstance(m, HumanMessage) else "assistant")
        api_messages.append({"role": role, "content": m.content})

      try:
        resp = openai.ChatCompletion.create(model="gpt-4o", messages=api_messages, max_tokens=500)
        content = resp["choices"][0]["message"]["content"]
        message_placeholder.info(content)
        st.session_state.messages.append(AIMessage(content=content))
      except Exception as e:
        st.error(f"OpenAI API call failed: {e}")
    else:
      try:
        for part in chat.stream(st.session_state.messages):
          stream = getattr(part, "content", "")
          if stream == " ":
            full_response += " "
          full_response = full_response + stream
          time.sleep(0.05)
          message_placeholder.info(full_response)
          st.session_state.messages.append(AIMessage(content=stream))
      except Exception:
        # Fallback: single response
        resp = chat(st.session_state.messages)
        message_placeholder.info(resp.content)
        st.session_state.messages.append(AIMessage(content=resp.content))

  elif option == 'Student Evaluation':
    subject = st.selectbox('Select Subject:', ['Essay-Writing'])
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file:
      raw_text = read_pdf(uploaded_file)

      # Splitting up the text into smaller chunks for indexing
      text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
      )
      texts = text_splitter.split_text(raw_text)
      full_stream = "\nTHESIS CLARITY :  \n"
      embeddings = OpenAIEmbeddings()
      docsearch = FAISS.from_texts(texts, embeddings)
      try:
          chain = load_qa_chain(ChatOpenAI(temperature=0, openai_api_key=api_key), chain_type="stuff")
      except Exception as e:
        st.error(f"Failed to initialize QA chain: {e}")
        return
      query = (
        "evaluate the given message, Evaluation criteria: 1. Thesis clarity: Clear and focused thesis statement that addresses the essay prompt"
        " 2. Analysis Depth: In-depth analysis of the topic with supporting evidence 3. Organization: Well structured and logically organized essay"
        " 4. Writing Clarity: Clear and concise writing with proper grammar and punctuation 5. Conclusion"
      )
      message = st.empty()
      if query:
        docs = docsearch.similarity_search(query)
        stream = chain.run(input_documents=docs, question='write a Clear and focused thesis statement that addresses the essay')
        message.info(stream)

        sentence = chain.run(input_documents=docs, question='write an In-depth analysis of the topic with supporting evidence')
        st.info("ANALYSIS DEPTH:\n" + sentence)

        s = chain.run(input_documents=docs, question='write a Well structured and logically organized essay based on the topic of the pdf')
        st.info("ORGANIZED ESSAY:\n" + s)

        write = chain.run(input_documents=docs, question='see if the text contains any grammatical errors or not')
        st.info("WRITING CLARITY:\n" + write)

        strea = chain.run(input_documents=docs, question='write down a conclusion')
        st.info("CONCLUSION:\n" + strea)

  elif option == 'Essay-Writing-handwritten':
    uploaded_file = st.file_uploader("Choose a handwritten file", type=["jpg", "jpeg", "png"])
    if uploaded_file:
      # use `api_key` derived earlier from environment or Streamlit secrets

      def encode_image(image_file):
        return base64.b64encode(image_file.read()).decode('utf-8')

      base64_image = encode_image(uploaded_file)

      headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
      }

      payload = {
        "model": "gpt-4o",
        "messages": [
          {
            "role": "user",
            "content": [
              {"type": "text", "text": "check for any grammatical errors in the image"},
              {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
          }
        ],
        "max_tokens": 300
      }

      response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

      # Parse JSON safely
      try:
        response_dict = response.json()
      except Exception:
        st.error("Failed to parse response from OpenAI API")
        return

      # Extract the content from the assistant's message
      try:
        content = response_dict["choices"][0]["message"]["content"]
      except Exception:
        st.error("Unexpected API response format")
        return

      content_cleaned = content.replace("\\n", "\n")
      messages = st.empty()
      full_answer = ""
      for ch in content_cleaned:
        full_answer = full_answer + ch
        time.sleep(0.005)
        messages.info(full_answer)


if __name__ == '__main__':
  main()

