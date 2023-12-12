# 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
# Import os to set API key
import os

from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.agents.agent_toolkits import  create_vectorstore_agent, VectorStoreToolkit, VectorStoreInfo

import streamlit as st

os.environ['OPENAI_API_KEY'] = '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'

llm = OpenAI(temperature=0.1, verbose=True)
embeddings = OpenAIEmbeddings()

loader = PyPDFLoader('annualreport.pdf')
pages = loader.load_and_split()
store = Chroma.from_documents(pages, embeddings, collection_name='annualreport')

vectorstore_info = VectorStoreInfo(name="annual_report",description="a banking annual report as a pdf",vectorstore=store)
toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

agent_executor = create_vectorstore_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)
st.title('GPT Investment Banker')
prompt = st.text_input('Input your prompt here')

if prompt:
    response = agent_executor.run(prompt)
    st.write(response)

    with st.expander('Document Similarity Search'):
        search = store.similarity_search_with_score(prompt) 
        st.write(search[0][0].page_content) 