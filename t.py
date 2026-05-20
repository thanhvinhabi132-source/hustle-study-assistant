import streamlit as st  # type: ignore
from google import genai
from PyPDF2 import PdfReader
import json

# Các thư viện bổ sung cho cấu trúc dữ liệu RAG
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# --- CẤU HÌNH ---
# Gọi API Key một cách an toàn từ Secrets của Streamlit
MY_API_KEY = st.secrets["GEMINI_API_KEY"]

# Khởi tạo Client duy nhất cho SDK chính thức của Google GenAI
client = genai.Client(api_key=MY_API_KEY)

# Lớp tạo Vector tùy biến sử dụng trực tiếp API chính thống
class DirectGeminiEmbeddings(Embeddings):
    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            try:
                safe_text = text.encode('utf-8', 'ignore').decode('utf-8')
                if not safe_text.strip():
                    safe_text = "empty"
                response = client.models.embed_content(
                    model="text-embedding-004",
                    contents=safe_text
                )
                embeddings.append(response.embeddings[0].values)
            except Exception:
                embeddings.append([0.0] * 768)
        return embeddings

    def embed_query(self, text):
        try:
            safe_text = text.encode('utf-8', 'ignore').decode('utf-8')
            response = client.models.embed_content(
                model="text-embedding-004",
                contents=safe_text
            )
            return response.embeddings[0].values
        except Exception:
            return [0.0] * 768

# --- GIAO DIỆN ỨNG DỤNG (STREAMLIT) ---
st.set_page_config(page_title="HUSTle Assistant", page_icon="🎓", layout="centered")

st.title("🎓 HUSTle Study Assistant")
st.markdown("""
    *Hệ thống hỗ trợ học tập thông minh dành cho sinh viên Bách Khoa.*
    *Tải file PDF bài giảng lên để AI tóm tắt, đặt câu hỏi ôn tập và hỏi đáp siêu tốc.*
""")

# 1. Khu vực Upload file
uploaded_file = st.file_uploader("Chọn file PDF bài giảng (Slide, giáo trình...)", type="pdf")

if uploaded_file is not None:
    # Khóa v6 đồng bộ trạng thái lưu trữ mới sạch sẽ trên bộ nhớ RAM
    if "vector_db_v6" not in st.session_state:
        with st.status("Đang xây dựng cơ sở dữ liệu RAG...", expanded=True) as status:
            st.write("Đang trích xuất văn bản từ PDF...")
            reader = PdfReader(uploaded_file)
            full_text = ""
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
            
            full_text = full_text.encode('utf-8', 'ignore').decode('utf-8')
            
            if not full_text.strip():
                status.update(label="Lỗi: PDF không có dữ liệu văn bản!", state="error", expanded=True)
                st.stop()
                
            st.write("Đang cắt nhỏ tài liệu thành các phân đoạn (Chunks)...")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_text(full_text)
            docs = [Document(page_content=chunk) for chunk in chunks]
            
            st.write("Đang khởi tạo cấu trúc tìm kiếm FAISS bằng SDK trực tiếp...")
            try:
                embeddings_model = DirectGeminiEmbeddings()
                vector_db = FAISS.from_documents(docs, embeddings_model)
                
                st.session_state["vector_db_v6"] = vector_db
                st.session_state["full_text_backup"] = full_text
                status.update(label="Xử lý dữ liệu RAG thành công!", state="complete", expanded=False)
            except Exception as embed_err:
                # 🔥 ĐÃ SỬA LỖI: Điền đầy đủ chuỗi f-string không bị mất chữ
                status.update(label=f"Lỗi khi tạo Vector: {embed_err}", state="error", expanded=True)
                st.stop()

    # 2. Nút bấm phân tích
    if st.button("🚀 Bắt đầu phân tích với AI"):
        with st.spinner('Đợi chút, Gemini đang soạn đề giúp bạn...'):
            try:
                retriever = st.session_state["vector_db_v6"].as_retriever(search_kwargs={"k": 6})
                relevant_docs = retriever.invoke("khái niệm định nghĩa lý thuyết trọng tâm công thức bài tập")
                context_text = "\n---\n".join([doc.page_content for doc in relevant_docs])

                prompt_content = f"""
                Bạn là một giáo sư tại Đại học Bách Khoa Hà Nội. 
                Dựa trên nội dung tài liệu cốt lõi sau đây:
                ---
                {context_text}
                ---
                Hãy phân tích và trả về một chuỗi JSON duy nhất (không bọc trong câu chữ dẫn nhập, không bọc trong tag ```json) theo đúng cấu trúc:
                {{
                  "summary": ["Ý tóm tắt 1", "Ý tóm tắt 2", "Ý tóm tắt 3"],
                  "terms": [
                     {{"term": "Thuật ngữ chuyên ngành 1", "definition": "Giải thích định nghĩa 1"}}
                  ],
                  "quiz": [
                     {{
                       "question": "Nội dung câu hỏi trắc nghiệm 1?",
                       "options": ["A. Đáp án A", "B. Đáp án B", "C. Đáp án C", "D. Đáp án D"],
                       "correct": "A. Đáp án A",
                       "explain": "Lời giải thích ngắn gọn."
                     }}
                  ]
                }}
                Yêu cầu: Tạo đúng 5 câu hỏi trắc nghiệm trong mảng "quiz".
                """

                response = client.models.generate_content(
                    model="gemini-2.5-flash", 
                    contents=prompt_content
                )

                clean_text = response.text.strip()
                if clean_text.startswith("```json"):
                    clean_text = clean_text[7:]
                elif clean_text.startswith("```"):
                    clean_text = clean_text[3:]
                
                if clean_text.endswith("```"):
                    clean_text = clean_text[:-3]
                
                clean_text = clean_text.strip()

                parsed_json = json.loads(clean_text)
                st.session_state["ai_data"] = parsed_json
                st.success
