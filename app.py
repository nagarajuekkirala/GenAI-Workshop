from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

from dotenv import load_dotenv
import os
load_dotenv()

key = os.getenv('OPEN_API_KEY')

loader = PyPDFDirectoryLoader(r'D:\ML_DSNR')
doc = loader.load()


splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap = 200)
documents = splitter.split_documents(doc)

embedding = OpenAIEmbeddings(api_key=key)
vector_store = Chroma(embedding_function=embedding)
vector_store.add_documents(doc)
retriever = vector_store.as_retriever()

prompt = ChatPromptTemplate.from_template("""Based on user question give the correct answer based only on context
                                 question:{question}
                                 context:{context}""")

llm = ChatOpenAI(api_key=key,model='gpt-4o-mini',temperature=0)

rag_chain = ({'context':retriever,"question":RunnablePassthrough()} | prompt | llm | StrOutputParser() )

st.image(r'https://innomatics.in/wp-content/uploads/2023/01/innomatics-footer-logo.png')

query = st.text_input("Enter the Query")

if st.button("Predict"):
    resposne = rag_chain.invoke(query)
    st.write(resposne)