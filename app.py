import streamlit as st
from huggingface_hub import InferenceClient
import docx
import PyPDF2
from langchain.text_splitter import CharacterTextSplitter
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
import requests

def read_docx(file):
    doc = docx.Document(file)
    text = ""
    for para in doc.paragraphs:
        text += para.text
    return text

def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

def preprocess_and_vectorize(text, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    # Split the text into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = text_splitter.split_text(text)

    # Load pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Generate embeddings for each chunk
    embeddings = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        embeddings.append(embedding)

    embeddings = np.vstack(embeddings).astype('float32')

    # Initialize FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index, chunks, dimension

# Show title and description.
st.title("📄 Document Question Answering")
st.write(
    "Upload a document below and ask a question about it – the selected model will answer! "
    "To use this app, you need to provide a Hugging Face API key, which you can get [here](https://huggingface.co/settings/tokens). "
)

# Model selection
model_options = {
    "Mistral": "mistralai/Mistral-7B-Instruct-v0.2",
    "Gemma": "distilbert-base-uncased",
    "Zephyr": "bert-base-uncased"
}
model_name = st.selectbox("Select a model", list(model_options.keys()))
if model_name not in model_options:
    st.error("Invalid model selected.")
    st.stop()

# Ask user for their Hugging Face API key via `st.text_input`.
huggingface_api_key = st.text_input("Hugging Face API Key", type="password")
if not huggingface_api_key:
    st.info("Please add your Hugging Face API key to continue.", icon="🗝️")
    st.stop()

# Create a Hugging Face Inference API client.
try:
    client = InferenceClient(model=model_options[model_name], token=huggingface_api_key)
except requests.exceptions.HTTPError as e:
    if e.response.status_code == 404:
        st.error("Model repository not found. Please check the model name and try again.")
    else:
        st.error(f"An error occurred: {e}")
    st.stop()

# Let the user upload a file via `st.file_uploader`.
uploaded_file = st.file_uploader(
    "Upload a document (.txt, .md, .docx, or .pdf)", type=("txt", "md", "docx", "pdf")
)

# Ask the user for a question via `st.text_area`.
question = st.text_area(
    "Now ask a question about the document!",
    placeholder="Can you give me a short summary?",
)

if uploaded_file and question:
    if uploaded_file.type == "text/plain":
        document = uploaded_file.read().decode("utf-8")
    elif uploaded_file.type == "text/markdown":
        document = uploaded_file.read().decode("utf-8")
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        document = read_docx(uploaded_file)
    elif uploaded_file.type == "application/pdf":
        document = read_pdf(uploaded_file)
    else:
        st.error("Unsupported file type.")
        document = ""

    # Preprocess and vectorize the document
    index, chunks, dimension = preprocess_and_vectorize(document)

    # Generate query embedding using the same model
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    inputs = tokenizer(question, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    query_embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy().astype('float32')

    # Ensure the query embedding has the same dimension as the document embeddings
    if query_embedding.shape[1] != dimension:
        st.error("Dimension mismatch between query embedding and document embeddings.")
        st.stop()

    # Search the index for the most similar chunk
    D, I = index.search(query_embedding, 1)
    most_similar_chunk = chunks[I[0][0]]

    # Generate an answer using the Hugging Face Inference API.
    headers = {"Authorization": f"Bearer {huggingface_api_key}", "Content-Type": "application/json"}
    response = requests.post(
        f"https://api-inference.huggingface.co/models/{model_options[model_name]}",
        headers=headers,
        json={"inputs": f"question: {question}, context: {most_similar_chunk}"}
    )

    # Check for errors in the response
    if response.status_code != 200:
        st.error(f"Error: {response.status_code} - {response.text}")
    else:
        # Parse the JSON response
        response_json = response.json()

        # Print the entire JSON response for debugging
        st.write(response_json)
        
        # Check if the response is a list and contains the 'generated_text' key
        if isinstance(response_json, list) and 'generated_text' in response_json[0]:
            generated_text = response_json[0]['generated_text']
            
            # Extract the answer from the generated text
            answer_start = generated_text.find("Short summary: ")
            if answer_start != -1:
                answer = generated_text[answer_start + len("Short summary: "):]
                st.write(answer)
            else:
                st.write(generated_text)  # Display the entire generated text if no specific answer format is found
        else:
            st.error("The response does not contain a 'generated_text' key.")


# Suggestions for improving efficiency and runtime:
# 1. Cache the preprocessing and vectorization steps to avoid redundant computations.
# 2. Use a more efficient text splitting strategy to balance chunk size and overlap.
# 3. Optimize the document reading functions to handle large files more efficiently.
