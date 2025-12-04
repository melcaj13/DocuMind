import streamlit as st
from dotenv import load_dotenv
import os
from pypdf import PdfReader
import time
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from huggingface_hub import InferenceClient

st.set_page_config(
    page_title="DocuMind",
    page_icon="🦄",
    layout="wide"
)
st.markdown("""
<style>
    .stApp {
        background-image: linear-gradient(to top, #a18cd1 0%, #fbc2eb 100%);
        background-attachment: fixed;
    }
    [data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.6);
        border-right: 2px solid #fbc2eb;
    }
    [data-testid="stChatMessage"] {
        background-color: rgba(255, 255, 255, 0.30);
        border-radius: 20px;
        border: 1px solid #ffffff;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        padding: 15px;
    }
    .stButton>button {
        background: linear-gradient(45deg, #ff9a9e 0%, #fad0c4 99%, #fad0c4 100%);
        color: white;
        border: none;
        border-radius: 25px;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .stButton>button:hover {
        transform: scale(1.05);
        color: white;
    }
    h1, h2, h3 {
        color: #4a4a4a !important;
        font-family: 'Comic Sans MS', sans-serif;
    }
    .stChatInput textarea {
        background-color: rgba(255, 255, 255, 0.9) !important;
        border: 2px solid #fbc2eb !important; /* Pembe Kenarlık */
        color: #4a4a4a !important;
        border-radius: 20px !important;
    }
    .stChatInput textarea:focus {
        border-color: #ff9a9e !important; /* Koyu Pembe */
        box-shadow: 0 0 10px rgba(255, 154, 158, 0.5) !important; /* Pembe Işık */
    }
    .stChatInput button {
        color: #ff9a9e !important;
    }
</style>
""", unsafe_allow_html=True)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    vectorstore.save_local("faiss_index")

def get_answer_final(context, question):
    api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    client = InferenceClient(token=api_token)
    model_id = "HuggingFaceH4/zephyr-7b-beta"

    system_instruction = """You are a helpful AI assistant. 
    Use the provided 'Context' to answer the user's question.
    If the answer is not in the context, say "I cannot find the answer in the documents."
    Keep your answer concise and helpful."""

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
    ]
    
    try:
        response = client.chat_completion(
            model=model_id,
            messages=messages,
            max_tokens=512,
            temperature=0.3,
            top_p=0.9
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

def main():
    load_dotenv()
    
    col1, col2 = st.columns([1, 15])
    with col1:
        st.write("🦄") 
    with col2:
        st.title("DocuMind: Renkli Asistan")
    
    st.markdown("---")

    with st.sidebar:
        st.header("📂 Belge Merkezi")
        st.write("PDF'lerini buraya bırak, gerisini bana bırak! ✨")
        
        pdf_docs = st.file_uploader("PDF Yükle", accept_multiple_files=True)
        
        if st.button("Sihirli Değnekle Dokun (Analiz Et)", type="primary"):
            if not pdf_docs:
                st.warning("Henüz dosya seçmedin balım!")
            else:
                with st.spinner("✨ Yapay zeka tozları serpiştiriliyor..."):
                    raw_text = get_pdf_text(pdf_docs)
                    chunks = get_text_chunks(raw_text)
                    get_vector_store(chunks)
                    time.sleep(1) 
                    st.success("Hazır!")
                    st.balloons() 

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Sorunu buraya yazabilirsin..."):
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            model_kwargs = {'device': 'cpu'}
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs=model_kwargs
            )
            
            try:
                new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
                docs = new_db.similarity_search(prompt, k=3)
                context_text = " ".join([doc.page_content for doc in docs])
                
                with st.spinner("Düşünüyorum... 🌸"):
                    response = get_answer_final(context_text, prompt)
                    st.markdown(response)
                    
                    with st.expander("Kaynak Metni Gör"):
                        st.info(context_text)
                        
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                st.error("Önce PDF yükleyip butona basman lazım balım!")

if __name__ == '__main__':
    main()