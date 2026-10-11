%%writefile app.py
import streamlit as st
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ==========================================
# 1. System Constants & Enterprise UI Config
# ==========================================
st.set_page_config(page_title="EgSA AI Copilot | Egyptian Space Agency", page_icon="🛰️", layout="wide")

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stChatFloatingInputContainer {padding-bottom: 20px;}
    .stButton>button {border-radius: 6px; font-weight: bold; border: 1px solid #1E3A8A;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Secure Authentication Logic
# ==========================================
valid_users = {"admin": "admin123", "engineer": "EgSA2026"}

def check_password():
    def password_entered():
        if st.session_state["user_input"] in valid_users and st.session_state["pass_input"] == valid_users[st.session_state["user_input"]]:
            st.session_state["password_correct"] = True
            st.session_state["logged_in_user"] = st.session_state["user_input"]
            del st.session_state["pass_input"]
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/14/Egyptian_Space_Agency_logo.png/220px-Egyptian_Space_Agency_logo.png", width=150)
        st.title("EgSA Secure Portal 🔐")
        st.markdown("Enter your credentials to access the internal AI Copilot.")
        st.text_input("Username", key="user_input")
        st.text_input("Password", type="password", key="pass_input")
        st.button("Authenticate", on_click=password_entered, use_container_width=True)
        
        if "password_correct" in st.session_state and not st.session_state["password_correct"]:
            st.error("🚫 Access Denied: Invalid username or password.")
            
    return False

if not check_password():
    st.stop()

# ==========================================
# 3. High-Performance RAG System (Cached)
# ==========================================
@st.cache_resource(show_spinner=False)
def load_rag_system():
    # 1. Vector Store & Retriever
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-small", 
        model_kwargs={'device': 'cuda'}
    )
    db = Chroma(persist_directory="/kaggle/working/egsa_vector_db", embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 5}) # Balanced for speed and accuracy
    
    # 2. LLM Quantization Config
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_compute_dtype=torch.float16, 
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_use_double_quant=True
    )
    
    # 3. Model & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct", 
        quantization_config=quant_config, 
        device_map="auto"
    )
    
    # Force the model to stop exactly when the answer is done (Prevents hashtags/hallucinations)
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|im_end|>")
    ]
    
    # 4. Pipeline Optimization for SPEED and STRICTNESS
    pipe = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        max_new_tokens=384, 
        do_sample=False,        # Greedy decoding: FASTEST & strictly prevents hallucinations
        repetition_penalty=1.1, # Prevents word loops
        return_full_text=False,
        eos_token_id=terminators,
        pad_token_id=tokenizer.eos_token_id
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    
    # 5. Professional Bilingual Prompt
    prompt = PromptTemplate.from_template("""<|im_start|>system
You are a highly professional, strictly formal AI assistant for the Egyptian Space Agency (EgSA). 
You are an expert in space engineering, satellite operations, and space research.

CRITICAL RULES:
1. You MUST answer in the EXACT SAME LANGUAGE as the user's question (e.g., Arabic for Arabic, English for English).
2. Maintain a highly scientific, direct, and professional tone. NEVER use emojis, hashtags, or conversational fillers.
3. Base your answer ONLY on the provided Context. If the answer is not in the context, reply strictly with: "I don't have enough information based on the provided documents." (or its exact Arabic translation: "لا أملك معلومات كافية بناءً على الوثائق المتاحة.").

Context: 
{context}
<|im_end|>
<|im_start|>user
Question: {input}<|im_end|>
<|im_start|>assistant
""")
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
    rag_chain = ({"context": retriever | format_docs, "input": RunnablePassthrough()} | prompt | llm | StrOutputParser())
    return rag_chain

rag_chain = load_rag_system()

# ==========================================
# 4. Professional Sidebar & Dashboard
# ==========================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/14/Egyptian_Space_Agency_logo.png/220px-Egyptian_Space_Agency_logo.png", width=150)
    st.title("EgSA AI Copilot")
    st.success(f"User: {st.session_state['logged_in_user'].upper()} 👨‍🚀")
    st.markdown("---")
    st.info("Status: Secure Connection 🟢")
    st.markdown("**Core Developer:** Ahmed Ehab")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("Logout", use_container_width=True):
            st.session_state["password_correct"] = False
            st.rerun()

# ==========================================
# 5. Main Chat Interface
# ==========================================
st.title("EgSA Technical AI Assistant 🚀")
st.markdown("Welcome to the secure environment. You can query satellite manuals, space laws, and research papers (e.g., EgyptSat-1 performance studies).")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle new user input
if prompt := st.chat_input("Enter your query here... (e.g., What are the ML models used in the EgyptSat-1 study?)"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Processing documents & formulating response... 🛰️"):
            # Generate response
            response = rag_chain.invoke(prompt)
            # Cleanup any weird artifacts just in case
            response = response.replace("<|im_end|>", "").strip()
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})