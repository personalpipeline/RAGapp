import os
import warnings
from dotenv import load_dotenv

# ### MODIFICATION ###: Import necessary libraries
from ibm_watsonx_ai.foundation_models import ModelInference # Assuming this is still needed directly
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames # Assuming this is still needed directly

from langchain_ibm import WatsonxLLM, WatsonxEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA

import gradio as gr

# --- Configuration ---
warnings.filterwarnings('ignore')

# ### MODIFICATION ###: Load environment variables from .env file
load_dotenv()

# ### MODIFICATION ###: Get credentials from environment variables
ibm_api_key = os.getenv("IBM_API_KEY")
ibm_project_id = os.getenv("IBM_PROJECT_ID") # Ensure this is set in your .env
ibm_url = os.getenv("IBM_URL", "https://us-south.ml.cloud.ibm.com") # Default URL if not in .env

# ### MODIFICATION ###: Validate required variables
if not ibm_api_key:
    raise ValueError("API key 'IBM_API_KEY' not found in environment variables. Please set it in the .env file.")
if not ibm_project_id:
    raise ValueError("Project ID 'IBM_PROJECT_ID' not found in environment variables. Please set it in the .env file.")


# --- LLM Function ---
def get_llm():
    model_id = 'mistralai/mixtral-8x7b-instruct-v01'
    parameters = {
        GenParams.MAX_NEW_TOKENS: 256,
        GenParams.TEMPERATURE: 0.5,
    }
    # ### MODIFICATION ###: Initialize WatsonxLLM with credentials from environment
    # NOTE: Check langchain-ibm documentation for the correct parameter names
    # It might be 'apikey', 'credentials', etc. depending on version.
    watsonx_llm = WatsonxLLM(
        model_id=model_id,
        url=ibm_url,         # Use URL from env/default
        project_id=ibm_project_id, # Use Project ID from env
        apikey=ibm_api_key,   # Use API Key from env (Verify parameter name)
        params=parameters,
    )
    return watsonx_llm

# --- Document Loading Function ---
def document_loader(file_obj):
    # Added check if file_obj is None
    if file_obj is None:
        return "Error: No file uploaded."
    try:
        # Use file_obj.name which Gradio provides as the temporary file path
        loader = PyPDFLoader(file_obj.name)
        loaded_document = loader.load()
        print(f"Loaded {len(loaded_document)} pages from PDF.") # Debug print
        return loaded_document
    except Exception as e:
        return f"Error loading document: {e}"

# --- Text Splitting Function ---
def text_splitter(data):
    if not isinstance(data, list) or not data:
         return "Error: Invalid data for text splitting."
    try:
        text_splitter_obj = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=50,
            length_function=len,
        )
        chunks = text_splitter_obj.split_documents(data)
        print(f"Split document into {len(chunks)} chunks.") # Debug print
        return chunks
    except Exception as e:
        return f"Error splitting text: {e}"

# --- Embedding Model Function ---
def watsonx_embedding():
    embed_params = {
        # EmbedTextParamsMetaNames are likely related to direct SDK calls,
        # check if they apply when using WatsonxEmbeddings or if params are simpler
        # EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3, # Example, verify necessity
        # EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True}, # Example, verify necessity
    }
    # ### MODIFICATION ###: Initialize WatsonxEmbeddings with credentials from environment
    # NOTE: Check langchain-ibm documentation for the correct parameter names
    watsonx_embedding_model = WatsonxEmbeddings(
        model_id="ibm/slate-125m-english-rtrvr",
        url=ibm_url,         # Use URL from env/default
        project_id=ibm_project_id, # Use Project ID from env
        apikey=ibm_api_key,   # Use API Key from env (Verify parameter name)
        params=embed_params, # Verify if params are needed/correct here
    )
    return watsonx_embedding_model

# --- Vector Database Function ---
def vector_database(chunks):
    if not isinstance(chunks, list) or not chunks:
         return "Error: Invalid chunks for vector database creation."
    try:
        embedding_model = watsonx_embedding()
        print("Creating in-memory vector database...") # Debug print
        # NOTE: This creates an IN-MEMORY Chroma DB every time.
        # For production, you'd persist it to disk and load/update.
        vectordb = Chroma.from_documents(chunks, embedding_model)
        print("Vector database created.") # Debug print
        return vectordb
    except Exception as e:
        return f"Error creating vector database: {e}"

# --- Retriever Function ---
# Combined document loading, splitting, embedding, and retriever creation
def process_document_and_create_retriever(file_obj):
    # Use a temporary variable to avoid overwriting file_obj
    current_file = file_obj
    if current_file is None:
        return "Error: No document uploaded for processing."
    try:
        print(f"Processing document: {current_file.name}")
        splits = document_loader(current_file)
        if isinstance(splits, str): # Check if document_loader returned an error string
            print(f"Document loading failed: {splits}")
            return splits

        chunks = text_splitter(splits)
        if isinstance(chunks, str): # Check if text_splitter returned an error string
             print(f"Text splitting failed: {chunks}")
             return chunks

        vectordb = vector_database(chunks)
        if isinstance(vectordb, str): # Check if vector_database returned an error string
            print(f"Vector DB creation failed: {vectordb}")
            return vectordb

        retriever_obj = vectordb.as_retriever()
        print("Retriever created successfully.")
        return retriever_obj
    except Exception as e:
        print(f"Error creating retriever: {e}")
        return f"Error creating retriever: {e}"


# --- QA Chain Function ---
# Modified to accept the retriever object directly
def run_qa_chain(retriever_obj, query):
    if isinstance(retriever_obj, str): # Handle error passed from retriever creation
        return f"Cannot run QA chain due to previous error: {retriever_obj}"
    if not query:
        return "Error: Please provide a query."

    try:
        print(f"Running QA chain with query: {query}")
        llm = get_llm()
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff", # Consider other chain types if context exceeds limit
            retriever=retriever_obj,
            return_source_documents=False # Set to True if you want to see source chunks
        )
        response = qa.invoke(query) # Use invoke for newer Langchain versions
        result = response.get('result', "Error: Could not parse response.") # Safely get result
        print(f"QA chain response: {result}")
        return result
    except Exception as e:
        print(f"Error in QA chain: {e}")
        return f"Error in QA chain: {e}"

# --- Gradio Interface Logic ---
# Use Gradio Blocks for more control over state (like storing the retriever)
with gr.Blocks() as rag_application:
    gr.Markdown("# RAG Chatbot\nUpload a PDF document and ask questions.")

    # Store the retriever object in state
    retriever_state = gr.State(None)

    with gr.Row():
        pdf_upload = gr.File(label="Upload PDF File", file_count="single", file_types=['.pdf'])
        process_btn = gr.Button("Process Document")

    query_input = gr.Textbox(label="Input Query", lines=2, placeholder="Type your question here...")
    submit_btn = gr.Button("Ask Question")
    output_display = gr.Textbox(label="Output")

    # Define actions
    process_btn.click(
        fn=process_document_and_create_retriever,
        inputs=[pdf_upload],
        outputs=[retriever_state, output_display] # Update state and show status/error in output
    ).then(
        lambda: "Document processed. Ready for questions.", outputs=output_display # Update status message on success
    )

    submit_btn.click(
        fn=run_qa_chain,
        inputs=[retriever_state, query_input],
        outputs=[output_display]
    )

# ### MODIFICATION ###: Launch Gradio for deployment
print("Launching Gradio Interface...")
# Bind to all interfaces (0.0.0.0) and port 7860 for access within the VM/network
# Make sure port 7860 is open in your GCP VM's firewall rules!
rag_application.launch(server_name="0.0.0.0", server_port=7860)