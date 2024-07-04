import streamlit as st
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
import tempfile

# Set up the Streamlit app
st.title("PDF Query App")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    with st.spinner("Processing...."):
    # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Load the PDF
        loader = UnstructuredFileLoader(tmp_file_path)
        documents = loader.load()

        # Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)

        # Create embeddings
        embeddings = OllamaEmbeddings(model="llama3")

        # Create a FAISS vector store
        db = FAISS.from_documents(texts, embeddings)

        # Create a retriever
        retriever = db.as_retriever(search_kwargs={"k": 3})

    # Set up the language model
    llm = Ollama(model="llama3")

    # Create a RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    # Query input
    query = st.text_input("Enter your question about the PDF:")

    if query:
        # Get the answer
        answer = qa_chain.run(query)

        # Display the answer
        st.write("Answer:", answer)

else:
    st.write("Please upload a PDF file to start querying.")