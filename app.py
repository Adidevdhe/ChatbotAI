from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint

import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()
os.environ['Hugging_Face_Token']=os.getenv("Hugging_Face_Token")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["Langchain_Api_Key"] = os.getenv("Langchain_Api_Key")
os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please response to the user queries"),
        ("user","Questions:{questions}")
    ]
)

#Streamlit framework

st.title('Langchain Demo with GROQ API')
input_text = st.text_input("Search the topic u want")
'''
repo_id="mistralai/Mistral-7B-Instruct-v0.3"
llm=HuggingFaceEndpoint(repo_id=repo_id,max_length=128,temperature=0.7,token='Hugging_Face_Token')
'''
llm = ChatGroq(model="llama3-8b-8192")

output_parser=StrOutputParser()

chain= prompt|llm|output_parser

if input_text:
    '''
    result = chain.invoke(input_text)
    st.write(result)
    '''
    st.write(chain.invoke({'questions':input_text}))