# Import os to set API key
import os
# Import OpenAI as main LLM service
from langchain.llms import OpenAI
# Bring in streamlit for UI/app interface
import streamlit as st

# Import PDF document loaders...there's other ones as well!
from langchain.document_loaders import PyPDFLoader
# Import chroma as the vector store
from langchain.vectorstores import Chroma

# Import vector store stuff
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)

api_key = st.text_input('Input your OpenAI API key here')
option = st.selectbox('Which document would you like to use?', [
                      'Binance Filing', 'Coinbase Filing'])

documents = {'Binance Filing': ['sec.pdf', 'a filing from the SEC against Binance'], 'Coinbase Filing': [
    'coinbase.pdf', 'a filing from the SEC against Coinbase']}

if api_key:
    os.environ['OPENAI_API_KEY'] = api_key
    llm = OpenAI(temperature=0, verbose=True)
    loader = PyPDFLoader(documents[option][0])
    pages = loader.load_and_split()
    store = Chroma.from_documents(pages, collection_name='sec-filing')

    # Create vectorstore info object - metadata repo?
    vectorstore_info = VectorStoreInfo(
        name="filing",
        description=documents[option][1],
        vectorstore=store
    )
    # Convert the document store into a langchain toolkit
    toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

    # Add the toolkit to an end-to-end LC
    agent_executor = create_vectorstore_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True
    )

    st.title('Ask me anything about the filings')
    # Create a text input box for the user
    prompt = st.text_input('Input your prompt here')

    if prompt:
        # Then pass the prompt to the LLM
        response = agent_executor.run(prompt)
        # ...and write it out to the screen
        st.write(response)

        # With a streamlit expander
        with st.expander('Document Similarity Search'):
            # Find the relevant pages
            search = store.similarity_search_with_score(prompt)
            # Write out the first
            st.write(search[0][0].page_content)
else:
    st.warning('Please enter your OpenAI API key to use the app.')

    st.title('GPT SEC Filing')
    st.warning('Please enter a prompt to ask questions about the PDF.')
