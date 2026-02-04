import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

st.set_page_config(page_title="Semantic Question Matching Tool", layout="wide")

st.title("🔍 Tool So Sánh Ngữ Nghĩa Câu Hỏi Chatbot")

@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

model = load_model()

st.markdown("### 1️⃣ Upload file danh sách câu hỏi KB (Excel)")
uploaded_file = st.file_uploader("Upload file .xlsx", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    if "Câu hỏi" not in df.columns:
        st.error("File phải có cột tên là 'Câu hỏi'")
    else:
        kb_questions = df["Câu hỏi"].dropna().tolist()
        kb_embeddings = model.encode(kb_questions, convert_to_tensor=True)

        st.success(f"Đã load {len(kb_questions)} câu hỏi từ KB")

        st.markdown("### 2️⃣ Nhập câu hỏi cần kiểm tra")
        user_question = st.text_input("Nhập câu hỏi của user:")

        if user_question:
            user_embedding = model.encode(user_question, convert_to_tensor=True)
            cos_scores = util.cos_sim(user_embedding, kb_embeddings)[0]

            results = []
            for i, score in enumerate(cos_scores):
                results.append((kb_questions[i], float(score)))

            results = sorted(results, key=lambda x: x[1], reverse=True)

            st.markdown("### 📊 Kết quả tương đồng cao nhất")
            for q, score in results[:5]:
                st.write(f"**{round(score*100,2)}%** — {q}")


