from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()
key = os.getenv('OPENAI_API_KEY')

question = st.text_area("Enter the Question: ")

model = ChatOpenAI(api_key=key,
                   model = 'gpt-4.1-mini',
                   temperature=0.5)

prompt = ChatPromptTemplate(("human", """You are given a user question. Your tasks:

1. Classify the question into exactly one of the following categories:
   - Python
   - SQL
   - Power BI
   - Machine Learning

2. Extract the most important keywords from the question.
   - Keywords should be relevant technical terms, functions, concepts, or operations.
   - Provide 3–4 keywords depending on question length.

3. If unclear, choose the closest matching category based on context.

Return the output in the following JSON format:

{{
  "category": "<Python | SQL | Power BI | Machine Learning |Large Language Model |Agents>",
  "keywords": ["keyword1", "keyword2", ...],
   'Answer': 'Explain about question in 100-200 tokens  ',
    'Sample Code': 'Explain simple python code snippet',
   'Important Links': 'Provide research papers/youtube links'                                   
                                           
}}

Here is the question:
{question}
""")
    )

chain = prompt | model

if st.button("Submit"):
    response = chain.invoke({'question':question})
    st.json(response.content)