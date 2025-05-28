import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA
# Define color palette with improved contrast
primary_color = "#007BFF"  # Bright blue for primary buttons
secondary_color = "#FFC107"  # Amber for secondary buttons
background_color = "#F8F9FA"  # Light gray for the main background
sidebar_background = "#2C2F33"  # Dark gray for sidebar (better contrast)
text_color = "#212529"  # Dark gray for content text
sidebar_text_color = "#FFFFFF"  # White text for sidebar
header_text_color = "#000000"  # Black headings for better visibility

st.markdown("""
    <style>
    /* Main Background */
    .stApp {{
        background-color: #F8F9FA;
        color: #212529;
    }}

    /* Sidebar Styling */
    [data-testid="stSidebar"] {{
        background-color: #2C2F33 !important;
        color: #FFFFFF !important;
    }}
    [data-testid="stSidebar"] * {{
        color: #FFFFFF !important;
        font-size: 16px !important;
    }}

    /* Headings */
    h1, h2, h3, h4, h5, h6 {{
        color: #000000 !important;
        font-weight: bold;
    }}

    /* Fix Text Visibility */
    p, span, div {{
        color: #212529 !important;
    }}

    /* File Uploader */
    .stFileUploader>div>div>div>button {{
        background-color: #FFC107;
        color: #000000;
        font-weight: bold;
        border-radius: 8px;
    }}

    /* Fix Navigation Bar (Top Bar) */
    header {{
        background-color: #1E1E1E !important;
    }}
    header * {{
        color: #FFFFFF !important;
    }}
    </style>
""", unsafe_allow_html=True)


# App title
st.title("📄 Build a RAG System with DeepSeek R1 & Ollama")

# Sidebar for instructions and settings
with st.sidebar:
    st.header("Instructions")
    st.markdown("""
    1. Upload a PDF file using the uploader below.
    2. Ask questions related to the document.
    3. The system will retrieve relevant content and provide a concise answer.
    """)

    st.header("Settings")
    st.markdown("""
    - **Embedding Model**: HuggingFace
    - **Retriever Type**: Similarity Search
    - **LLM**: DeepSeek R1 (Ollama)
    """)

st.title ("Rag Agent using Deepseek R1 and Ollama ")

uploaded_file_document=st.file_uploader("Upload a PDF document for Analysis", type="pdf")

if uploaded_file_document: 
    with open("ragtemp.pdf", "wb") as f:
        f.write(uploaded_file_document.getvalue())
    
    load_pdf = PDFPlumberLoader("ragtemp.pdf")
    docs = load_pdf.load()

    text_doc_splitter = SemanticChunker(HuggingFaceEmbeddings())
    documents = text_doc_splitter.split_documents(docs)

    hugging_embedder = HuggingFaceEmbeddings()
    vector=FAISS.from_documents(documents, hugging_embedder)
    retriver = vector.as_retriever(search_type ='similarity',search_kwargs={"k": 3})

    llm_model = Ollama(model="deepseek-r1:1.5b")

    prompt= """
            Please use the followng context to answer the question.
            Context: {context}
            Question: {question}
            Answer:"""
    QA_Prompt = PromptTemplate.from_template(prompt)

    llm_chain = LLMChain(llm=llm_model, prompt=QA_Prompt,verbose=True)

    documents_prompt = PromptTemplate(
        input_variables=["page_content", "source"],
        template = "Context:\ncontent: {page_content}\nSource: {source}"
    )

    combine_documents_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context",
        document_prompt=documents_prompt,
        verbose=True
    )

    qa_retriver = RetrievalQA.from_chain_type(
    llm=llm_model,
    retriever=retriver,
    chain_type="stuff",
    chain_type_kwargs={"prompt": QA_Prompt},
    return_source_documents=True,
    verbose=True
    )

    
    st.header("Ask a Question?")

    user_question = st.text_input("Enter your question here related to the document:")

    if user_question:
        with st.spinner("Processing your question..."):
            try:
                response = qa_retriver(user_question)["result"]
                st.success("Response success")
                st.write(response)
            except Exception as e:
                st.error(f"An Exception occurred: {e}")
    else:
        st.info("Please uopload a PDF document and enter a question to get started.")