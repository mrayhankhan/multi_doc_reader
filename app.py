import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.modeling_utils")

import os
import sys
import subprocess
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, pipeline
import requests

# Load environment variables
load_dotenv()
os.environ["HUGGING_FACE_HUB_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Set TensorFlow environment variable to disable oneDNN custom operations
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=10,
        length_function=len
    )
    return splitter.split_text(text)

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_texts(text_chunks, embeddings)
    return vectorstore

def get_conversation_chain(vectorstore, model, tokenizer):
    memory = ConversationBufferMemory()
    chain = ConversationalRetrievalChain(
        retriever=vectorstore.as_retriever(),
        memory=memory,
        llm=model,
        tokenizer=tokenizer
    )
    return chain

def handle_userinput(user_question, model_name):
    if model_name == "mistral":
        api_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-Nemo-Instruct-2407"
    elif model_name == "gemma":
        api_url = "https://api-inference.huggingface.co/models/google/gemma-7b"
    elif model_name == "zephyr":
        api_url = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-alpha"
    else:
        st.error("Invalid model selected")
        return

    headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACEHUB_API_TOKEN')}"}
    payload = {"inputs": user_question}

    response = requests.post(api_url, headers=headers, json=payload)
    response_json = response.json()

    if response.status_code == 200:
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        st.session_state.chat_history.append({"role": "bot", "content": response_json[0]['generated_text']})

        for i, message in enumerate(st.session_state.chat_history):
            if message["role"] == "user":
                st.write(user_template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)
    else:
        st.error("Error: " + response_json.get("error", "Unknown error"))

def main():
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if 'conv' not in st.session_state:
        st.session_state.conv = []
    if 'robo' not in st.session_state:
        st.session_state.robo = ['How can I help you?']

    st.header("Chat with multiple PDFs :books:")

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process PDFs"):
            with st.spinner("Processing PDFs"):
                # Step 1: Get PDF text
                raw_text = get_pdf_text(pdf_docs)

                # Step 2: Get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # Step 3: Create vector store
                vectorstore = get_vectorstore(text_chunks)

                st.session_state.vectorstore = vectorstore
                st.success("PDFs processed successfully!")

    if "vectorstore" in st.session_state:
        model_option = st.selectbox("Select Model", ["Gemma", "Mistral", "Zephyr"])
        if st.button("Load Model"):
            with st.spinner("Loading model"):
                if model_option == "Gemma":
                    config = AutoConfig.from_pretrained("google/gemma-7b")
                    config.hidden_activation = "gelu_pytorch_tanh"  # Set your desired activation function here
                    gemma_tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b", token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))
                    gemma_model = AutoModelForCausalLM.from_pretrained("google/gemma-7b", config=config, token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))
                    model = gemma_model
                    tokenizer = gemma_tokenizer
                elif model_option == "Mistral":
                    mistral_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-Nemo-Instruct-2407", token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))
                    mistral_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-Nemo-Instruct-2407", token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))
                    model = mistral_model
                    tokenizer = mistral_tokenizer
                else:
                    zephyr_tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-alpha", token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))
                    zephyr_model = AutoModelForCausalLM.from_pretrained("HuggingFaceH4/zephyr-7b-alpha", token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))
                    model = zephyr_model
                    tokenizer = zephyr_tokenizer

                st.session_state.conversation = get_conversation_chain(st.session_state.vectorstore, model, tokenizer)
                st.success("Model loaded successfully!")

    if st.session_state.conversation:
        user_question = st.text_input("Ask a question about your documents:")
        if user_question:
            handle_userinput(user_question, model_option.lower())

if __name__ == "__main__":
    # Check if the script is being run by Streamlit
    if not os.getenv("STREAMLIT_RUN"):
        os.environ["STREAMLIT_RUN"] = "true"
        subprocess.run(["streamlit", "run", sys.argv[0]])
    else:
        main()
