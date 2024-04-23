import streamlit as st
import os
from PyPDF2 import PdfReader
import docx
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from streamlit_chat import message
from langchain_community.callbacks import get_openai_callback

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat With any files")
    st.header("💬 Chatbot 🤖")

    st.session_state.conversation = None
    st.session_state.chat_history = None
    st.session_state.processComplete = True  # Directly set to True to skip processing

    pdf_filename = "MaternatiyLaw.pdf"  # PDF file to be used directly

    text_chunks = get_text_chunks(get_pdf_text(pdf_filename))
    vector_store = get_vectorstore(text_chunks)
    api_key = "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"  # Your OpenAI API key
    st.session_state.conversation = get_conversation_chain(vector_store, api_key)

    user_question = st.text_input("Ask Question to the Chatbot.")
    if user_question:
        st.text("Bot is typing... 🤖⌨")
        handel_userinput(user_question)

def get_pdf_text(pdf_filename):
    pdf_reader = PdfReader(pdf_filename)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=900,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings()
    knowledge_base = FAISS.from_texts(text_chunks, embeddings)
    return knowledge_base

def get_conversation_chain(vector_store, openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name='gpt-3.5-turbo', temperature=0)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handel_userinput(user_question):
    with get_openai_callback() as cb:
        response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    response_container = st.container()
    with response_container:
        for i, messages in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.text("🤖: " + messages.content)  # Emojis for chatbot responses
            else:
                st.text("You: " + messages.content)  # Display user messages

if __name__ == "__main__":
    main()
