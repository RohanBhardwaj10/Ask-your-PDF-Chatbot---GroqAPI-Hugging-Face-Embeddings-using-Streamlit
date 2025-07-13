import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os

from dotenv import load_dotenv
load_dotenv()

os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

api_key = os.getenv("GROQ_API_KEY")
st.title("Ask my PDF ! Chatbot with chat history ")
st.write("Upload Pdf's and chat with their content")
model = ChatGroq(model_name="Gemma2-9b-It",groq_api_key=api_key)
session_id = "abc123"

if 'store' not in st.session_state:
        st.session_state.store={} 
uploaded_files=st.file_uploader("Choose a PDF file",type="pdf",accept_multiple_files=True)
if uploaded_files:
    documents=[]
    for files in uploaded_files:
        temppdf=f"./temp.pdf"
        with open(temppdf,'wb') as file:
            file.write(files.getvalue())
            file_name=files.name
        loader=PyPDFLoader("temp.pdf")
        docs=loader.load()
        documents.extend(docs)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever() 
    contextualize_q_system_prompt=(
            "Given a chat history and the latest user question"
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
    contextualize_q_prompt= ChatPromptTemplate.from_messages(
        [
            ("system",contextualize_q_system_prompt),
            ("human","{input}"),
        ]
    )
    history_aware_retriever=create_history_aware_retriever(model,retriever,contextualize_q_prompt)
    system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know. Use three sentences maximum and keep the "
                "answer concise."
                "\n\n"
                "{context}"
            )
    qa_prompt= ChatPromptTemplate(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
 
    question_answer_chain=create_stuff_documents_chain(model,qa_prompt)
    rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)
    def get_session_history(session:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]
        
    conversational_rag_chain=RunnableWithMessageHistory(
            rag_chain,get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
    )
        
    user_input = st.text_input("Your question:")
    if user_input:
        session_history=get_session_history(session_id)
        response = conversational_rag_chain.invoke(
            {"input": user_input},
                config={
                    "configurable": {"session_id":"abc123"}
                },  
            )
        st.write("Assistant:", response['answer'])
