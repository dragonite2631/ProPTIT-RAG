import streamlit as st
import base64
import random
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from reranker_search import Reranker
from llm_api import LLM
from llm_local import LLM as LLM_local
from hybrid_search import HybridSearch

# --- PHẦN CẤU HÌNH TRANG VÀ BIỂU TƯỢNG TIÊU ĐỀ ---

# Đọc và mã hóa ảnh sang Base64
image_path = "Img/logopro.png"  # Đảm bảo đường dẫn này đúng
encoded_string = None
try:
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
except FileNotFoundError:
    print(f"LỖI: Không tìm thấy tệp tại đường dẫn '{image_path}'. Vui lòng kiểm tra lại.")

ICON_DATA_URL = f"data:image/png;base64,{encoded_string}"
ICON_PATH = "Img/logopro.png"

# Cấu hình trang
st.set_page_config(
    page_title="ProPTIT Chatbot",
    page_icon=ICON_PATH,
    layout="centered"
)

# Hiển thị tiêu đề với icon
st.markdown(f"""
<h1 style="display: flex; align-items: center; justify-content: center;">
    <img src="{ICON_DATA_URL}" alt="Biểu tượng CLB" style="width: 32px; height: 32px; margin-right: 15px; vertical-align: middle;">
    ProPTIT Chatbot 💬
</h1>
""", unsafe_allow_html=True)

st.markdown("Chào mừng bạn đến với hệ thống hỏi đáp tự động của CLB! Hãy nhập câu hỏi của bạn về CLB để nhận thông tin.")

# --- DANH SÁCH CÂU HỎI GỢI Ý ---
SUGGESTED_QUESTIONS = [
    "CLB ProPTIT có những ban nào?",
    "Làm thế nào để trở thành thành viên của CLB?",
    "Lịch sinh hoạt của CLB như thế nào?",
    "ProPTIT có những hoạt động ngoại khóa nào?",
    "Mình có cần biết lập trình trước khi tham gia không?",
    "Quyền lợi khi trở thành thành viên CLB là gì?"
]

# --- CÁC HÀM LOAD MODEL (Không thay đổi) ---
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

# --- GIAO DIỆN CHÍNH ---

# Khởi tạo lịch sử chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar với các gợi ý và cài đặt
with st.sidebar:
    use_rag_search = st.toggle("Sử dụng tìm kiếm tài liệu 🔍", value=True, key="use_rag_toggle")
    
    st.subheader("💡 Gợi ý cho bạn:")
    # Khởi tạo câu hỏi gợi ý nếu chưa có
    if "suggested_questions_this_session" not in st.session_state:
        st.session_state.suggested_questions_this_session = random.sample(SUGGESTED_QUESTIONS, 3)
    
    # Khi một nút được nhấp, lưu câu hỏi vào một key TẠM THỜI
    for question in st.session_state.suggested_questions_this_session:
        if st.button(question, key=f"btn_{question}", use_container_width=True):
            st.session_state.clicked_query = question # Sử dụng key khác, không phải "user_query"
            st.rerun()

# Hiển thị trạng thái tìm kiếm
if use_rag_search:
    st.info("Tìm kiếm đang được bật ✅")
else:
    st.warning("Tìm kiếm thông minh đang bị tắt ❌")

# Hiển thị lịch sử chat
for message in st.session_state.chat_history:
    avatar = "🤖" if message["role"] == "assistant" else "👤"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# --- LOGIC XỬ LÝ TRUY VẤN (Đã sửa lỗi) ---
user_query = None

# Ưu tiên xử lý câu hỏi từ nút bấm trước
if "clicked_query" in st.session_state and st.session_state.clicked_query:
    user_query = st.session_state.clicked_query
    del st.session_state.clicked_query # Xóa key tạm thời sau khi lấy giá trị

# Lấy câu hỏi từ ô chat input
if prompt := st.chat_input("Nhập câu hỏi của bạn về CLB...❓"):
    user_query = prompt

# Nếu có câu hỏi từ một trong hai nguồn, thì xử lý
if user_query:
    # Thêm câu hỏi của người dùng vào lịch sử
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    # Hiển thị tin nhắn người dùng ngay lập tức
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_query)

    # Xử lý logic chatbot
    with st.spinner(text="Đang suy nghĩ...",show_time=True):
        try:
            if use_rag_search: 
                doc_ids = hybrid_search.search(user_query, 60, 7, 7)
                prompt_for_llm = llm_model.generate_answer_prompt(user_query, list(doc_ids.keys()))
                ai_answer = llm_model.generate_response(prompt_for_llm, st.session_state.chat_history)
            else: 
                ai_answer = llm_model.generate_response(user_query, st.session_state.chat_history)
            
            # Thêm câu trả lời của bot vào lịch sử
            st.session_state.chat_history.append({"role": "assistant", "content": ai_answer})

        except Exception as e:
            error_message = f"Đã xảy ra lỗi trong quá trình xử lý: {e}"
            st.error(error_message)
            st.session_state.chat_history.append({"role": "assistant", "content": error_message})
    
    # Chạy lại toàn bộ script để hiển thị câu trả lời mới của bot
    st.rerun()
