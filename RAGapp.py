from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from ibm_watsonx_ai import Credentials
from langchain_ibm import WatsonxLLM, WatsonxEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA

import gradio as gr

import warnings
warnings.filterwarnings('ignore')

## LLM
def get_llm():
    model_id = 'mistralai/mixtral-8x7b-instruct-v01'
    parameters = {
        GenParams.MAX_NEW_TOKENS: 256,
        GenParams.TEMPERATURE: 0.5,
    }
    project_id = "skills-network"
    watsonx_llm = WatsonxLLM(
        model_id=model_id,
        url="https://us-south.ml.cloud.ibm.com",
        project_id=project_id,
        params=parameters,
    )
    return watsonx_llm

## Document loader
def document_loader(file_obj):
    try:
        loader = PyPDFLoader(file_obj.name)
        loaded_document = loader.load()
        return loaded_document
    except Exception as e:
        return f"Error loading document: {e}"

## Text splitter
def text_splitter(data):
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=50,
            length_function=len,
        )
        chunks = text_splitter.split_documents(data)
        return chunks
    except Exception as e:
        return f"Error splitting text: {e}"

## Embedding model
def watsonx_embedding():
    embed_params = {
        EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
        EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
    }
    watsonx_embedding_model = WatsonxEmbeddings(
        model_id="ibm/slate-125m-english-rtrvr",
        url="https://us-south.ml.cloud.ibm.com",
        project_id="skills-network",
        params=embed_params,
    )
    return watsonx_embedding_model

## Vector db
def vector_database(chunks):
    try:
        embedding_model = watsonx_embedding()
        vectordb = Chroma.from_documents(chunks, embedding_model)
        return vectordb
    except Exception as e:
        return f"Error creating vector database: {e}"

## Retriever
def retriever(file_obj):
    try:
        splits = document_loader(file_obj)
        if isinstance(splits, str):
            return splits
        chunks = text_splitter(splits)
        if isinstance(chunks, str):
            return chunks
        vectordb = vector_database(chunks)
        if isinstance(vectordb, str):
            return vectordb
        retriever_obj = vectordb.as_retriever()
        return retriever_obj
    except Exception as e:
        return f"Error creating retriever: {e}"

## QA Chain
def retriever_qa(file_obj, query):
    try:
        llm = get_llm()
        retriever_obj = retriever(file_obj)
        if isinstance(retriever_obj, str):
            return retriever_obj
        qa = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever_obj,
                                        return_source_documents=False)
        response = qa.invoke(query)
        return response['result']
    except Exception as e:
        return f"Error in QA chain: {e}"

# Create Gradio interface
rag_application = gr.Interface(
    fn=retriever_qa,
    allow_flagging="never",
    inputs=[
        gr.File(label="Upload PDF File", file_count="single", file_types=['.pdf']),
        gr.Textbox(label="Input Query", lines=2, placeholder="Type your question here...")
    ],
    outputs=gr.Textbox(label="Output"),
    title="RAG Chatbot",
    description="Upload a PDF document and ask any question. The chatbot will try to answer using the provided document."
)

rag_application.launch(share=True)