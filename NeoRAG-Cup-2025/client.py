from reranker_search import Reranker
from llm_api import LLM
from llm_local import LLM as LLM_local
from transformers import AutoModelForSequenceClassification,  AutoTokenizer
from hybrid_search import HybridSearch
import pandas as pd
import streamlit as st

@st.cache_resource 
def load_local_model():
    model_name = "Qwen/Qwen3-0.6B"
    base_data_path = "CLB_PROPTIT.csv"
    model = LLM_local(model_name, base_data_path)
    return model

@st.cache_resource
def load_model():
    llm_name = "openai/gpt-oss-20b"
    base_data_path = "CLB_PROPTIT.csv"
    llm_model = LLM(llm_name, base_data_path)
    return llm_model

llm_model = load_model()

@st.cache_resource
def load_hybrid_search(local):
    base_data_path = "CLB_PROPTIT.csv"
    train_data_path  = "train_data_proptit.xlsx"
    reranker_path ="Vietnamese_Reranker_finetuned"
    if local: 
        llm_name = "Qwen/Qwen3-0.6B"
        llm = LLM_local(llm_name, base_data_path)
    else:
        llm_name = "openai/gpt-oss-20b"
        llm = LLM(llm_name, base_data_path)
    bm25_type = "BM25L"
    reranker_model =Reranker(AutoModelForSequenceClassification.from_pretrained(reranker_path),
                            AutoTokenizer.from_pretrained(reranker_path),
                            base_data_path
                            )

    hybrid_search = HybridSearch(bm25_type, reranker_model, llm, base_data_path, train_data_path)
    return hybrid_search

hybrid_search = load_hybrid_search(local=False)

st.title("💬 Chatbot Hỏi Đáp CLB Lập Trình")
st.markdown("Chào mừng bạn đến với hệ thống hỏi đáp tự động của CLB! Hãy nhập câu hỏi của bạn về CLB để nhận thông tin.")
with st.sidebar:
    use_rag_search = st.toggle("Sử dụng tìm kiếm tài liệu", value=True, key="use_rag_toggle")
if use_rag_search:
    st.info("Tìm kiếm đang được bật")
else:
    st.warning("Tìm kiếm thông minh đang bị tắt")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
user_query = st.chat_input("Nhập câu hỏi của bạn về CLB...")
if user_query:
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    # print(type(st.session_state.chat_history))
    ch = st.session_state.chat_history
    with st.chat_message("user"):
        st.markdown(user_query)
    with st.spinner(text="Đang suy nghĩ...", show_time=True):
        try:
            
            if use_rag_search: 
                doc_ids = hybrid_search.search(user_query,60,7,7)
                print(doc_ids)
                prompt = llm_model.generate_answer_prompt(user_query,list(doc_ids.keys()))
                ai_answer = llm_model.generate_response(prompt, ch)
            else: 
                ai_answer = llm_model.generate_response(user_query, ch)
            st.session_state.chat_history.append({"role": "assistant", "content": ai_answer})
            with st.chat_message("assistant"):
                st.markdown(ai_answer)

        except Exception as e:
            error_message = f"Đã xảy ra lỗi trong quá trình xử lý: {e}"
            st.error(error_message)
            st.session_state.chat_history.append({"role": "assistant", "content": error_message})
            with st.chat_message("assistant"):
                st.markdown(error_message)