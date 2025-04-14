import os
import warnings
from dotenv import load_dotenv

try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    print("Swapped standard sqlite3 with pysqlite3.")
except ImportError:
    print("pysqlite3 not found or import error, using standard sqlite3.")
    # Depending on your requirements, you might want to raise an error here
    # if pysqlite3 is strictly necessary for ChromaDB to function correctly.
    pass

# ### MODIFICATION ###: Import Google Vertex AI components
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA

import gradio as gr

# --- Configuration ---
warnings.filterwarnings('ignore')

# ### MODIFICATION ###: Load environment variables
load_dotenv()

# ### MODIFICATION ###: Get Google Cloud Project ID from environment
google_cloud_project = os.getenv("GOOGLE_CLOUD_PROJECT")
if not google_cloud_project:
    raise ValueError("Project ID 'GOOGLE_CLOUD_PROJECT' not found in environment variables. Please set it in the .env file or system environment.")

# Note: Authentication is handled via GOOGLE_APPLICATION_CREDENTIALS env var pointing to your service account key file.

# --- LLM Function ---
def get_llm():
    model_name = "gemini-1.0-pro" # Replace if needed
    llm = VertexAI(
        model_name=model_name,
        project=google_cloud_project, # ### MODIFICATION ###: Pass Project ID
        temperature=0.5,
        max_output_tokens=256,
    )
    print(f"Initialized Vertex AI LLM: {model_name} in project {google_cloud_project}")
    return llm

# --- Document Loading Function ---
def document_loader(file_obj):
    if file_obj is None: return "Error: No file uploaded."
    try:
        loader = PyPDFLoader(file_obj.name)
        loaded_document = loader.load()
        print(f"Loaded {len(loaded_document)} pages from PDF.")
        return loaded_document
    except Exception as e: return f"Error loading document: {e}"

# --- Text Splitting Function ---
def text_splitter(data):
    if not isinstance(data, list) or not data: return "Error: Invalid data for text splitting."
    try:
        text_splitter_obj = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50, length_function=len)
        chunks = text_splitter_obj.split_documents(data)
        print(f"Split document into {len(chunks)} chunks.")
        return chunks
    except Exception as e: return f"Error splitting text: {e}"

# --- Inside get_vertex_embedding() ---
def get_vertex_embedding():
    # ### MODIFICATION ###: Try a different model name
    # model_name = "textembedding-gecko@003" # Original problematic one

    # --- OPTION A: Try the newer model ---
    model_name = "text-embedding-004"
    # --- OR OPTION B: Try an older stable gecko ---
    # model_name = "textembedding-gecko@001"

    embeddings = VertexAIEmbeddings(
        model_name=model_name,
        project=google_cloud_project, # Keep passing project ID
    )
    print(f"Initialized Vertex AI Embeddings: {model_name} in project {google_cloud_project}")
    return embeddings

# --- Vector Database Function ---
def vector_database(chunks):
    if not isinstance(chunks, list) or not chunks: return "Error: Invalid chunks for vector database creation."
    try:
        embedding_model = get_vertex_embedding()
        print("Creating in-memory vector database...")
        vectordb = Chroma.from_documents(chunks, embedding_model)
        print("Vector database created.")
        return vectordb
    except Exception as e:
        # ### MODIFICATION ###: Print detailed error during vector DB creation
        print(f"Detailed error creating vector database: {e}")
        # Pass the error message up, which includes the project ID issue if that's the root cause
        return f"Error creating vector database: {e}"


# --- Retriever Function ---
def process_document_and_create_retriever(file_obj):
    current_file = file_obj
    if current_file is None: return "Error: No document uploaded for processing." , None
    try:
        print(f"Processing document: {current_file.name}")
        splits = document_loader(current_file)
        if isinstance(splits, str): print(f"Document loading failed: {splits}"); return splits, None

        chunks = text_splitter(splits)
        if isinstance(chunks, str): print(f"Text splitting failed: {chunks}"); return chunks, None

        vectordb = vector_database(chunks)
        if isinstance(vectordb, str): print(f"Vector DB creation failed: {vectordb}"); return vectordb, None # Pass up the specific error

        retriever_obj = vectordb.as_retriever()
        print("Retriever created successfully.")
        return "Document processed successfully. Ready for questions.", retriever_obj
    except Exception as e:
        error_msg = f"Error creating retriever: {e}"; print(error_msg); return error_msg, None


# --- QA Chain Function ---
def run_qa_chain(retriever_obj, query):
    if retriever_obj is None: return "Error: Document not processed yet or processing failed. Please upload and process a document first."
    if isinstance(retriever_obj, str): return f"Cannot run QA chain due to previous error: {retriever_obj}" # Should be caught by retriever_obj is None
    if not query: return "Error: Please provide a query."

    try:
        print(f"Running QA chain with query: {query}")
        llm = get_llm()
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever_obj, return_source_documents=False)
        response = qa.invoke(query)
        result = response.get('result', "Error: Could not parse response.")
        print(f"QA chain response: {result}")
        return result
    except Exception as e:
        error_msg = f"Error in QA chain: {e}"; print(error_msg)
        if "permission denied" in str(e).lower(): error_msg += " (Check service account permissions for Vertex AI API)"
        elif "quota" in str(e).lower(): error_msg += " (Check Vertex AI Quotas for your project)"
        elif "project" in str(e).lower(): error_msg += f" (Ensure project '{google_cloud_project}' is correct and Vertex AI API is enabled)" # Added project check hint
        return error_msg

# --- Gradio Interface Logic ---
with gr.Blocks() as rag_application:
    gr.Markdown("# RAG Application using Vertex AI\nUpload a PDF document,process it and ask questions. (Only PDF files supported)")
    retriever_state = gr.State(None)
    status_display = gr.Textbox(label="Status", interactive=False)
    with gr.Row():
        pdf_upload = gr.File(label="Upload only aPDF File", file_count="single", file_types=['.pdf'])
        process_btn = gr.Button("Process Document")
    query_input = gr.Textbox(label="Input Query", lines=2, placeholder="Type your question here...")
    submit_btn = gr.Button("Ask Question")
    output_display = gr.Textbox(label="Output", lines=5)

    process_btn.click(fn=process_document_and_create_retriever, inputs=[pdf_upload], outputs=[status_display, retriever_state])
    submit_btn.click(fn=run_qa_chain, inputs=[retriever_state, query_input], outputs=[output_display])

# --- Launch Gradio ---
print("Launching Gradio Interface...")
rag_application.launch(server_name="0.0.0.0", server_port=7860)