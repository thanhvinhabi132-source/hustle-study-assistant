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

# Lớp tạo Vector tùy biến sử dụng trực tiếp API chính thống không qua thư viện bọc lỗi
class DirectGeminiEmbeddings(Embeddings):
    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            try:
                safe_text = text.encode('utf-8', 'ignore').decode('utf-8')
                if not safe_text.strip():
                    safe_text = "empty"
                # Gọi trực tiếp qua API chính thức của SDK gốc
                response = client.models.embed_content(
                    model="text-embedding-004",
                    contents=safe_text
                )
                embeddings.append(response.embeddings[0].values)
            except Exception:
                # Trả về vector trống nếu phân đoạn văn bản bị lỗi đặc biệt
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
    # Đổi sang Key v6 để dọn sạch hoàn toàn bộ nhớ cache cũ
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
                st.success("Phân tích hoàn tất!")

            except Exception as e:
                st.error(f"Có lỗi xảy ra trong quá trình phân tích: {e}")

    # 3. Hiển thị kết quả
    if "ai_data" in st.session_state:
        data = st.session_state["ai_data"]
        st.divider()
        
        st.markdown("### 📝 1. Tóm tắt kiến thức cốt lõi")
        for point in data.get("summary", []):
            st.markdown(f"- {point}")
            
        st.markdown("### 🔍 2. Thuật ngữ chuyên ngành")
        terms_list = data.get("terms", [])
        if terms_list:
            num_cols = min(len(terms_list), 3)
            cols = st.columns(num_cols)
            for idx, t in enumerate(terms_list):
                col_idx = idx % num_cols
                with cols[col_idx]:
                    st.info(f"**{t.get('term')}**\n\n*{t.get('definition')}*")

        st.markdown("### 🧠 3. Thử thách trắc nghiệm ôn tập")
        quiz_list = data.get("quiz", []):
        if quiz_list:
            tabs = st.tabs([f"Câu {x+1}" for x in range(len(quiz_list))])
            for i, tab in enumerate(tabs):
                with tab:
                    item = quiz_list[i]
                    st.markdown(f"**Câu hỏi:** {item.get('question')}")
                    user_choice = st.radio("Chọn một đáp án đúng:", options=item.get("options", []), index=None, key=f"q_v6_{i}")
                    
                    if user_choice:
                        if user_choice == item.get("correct"):
                            st.success("🎉 Xuất sắc!")
                        else:
                            st.error(f"❌ Đáp án đúng là: {item.get('correct')}")
                        st.caption(f"💡 Phân tích: {item.get('explain')}")

        st.divider()
        st.markdown("### 💬 4. Hỏi đáp Siêu tốc về Tài liệu (RAG Q&A)")
        user_question = st.text_input("Nhập câu hỏi của bạn:", key="rag_query_input")
        
        if user_question:
            with st.spinner("Đang truy vấn dữ liệu..."):
                try:
                    db_retriever = st.session_state["vector_db_v6"].as_retriever(search_kwargs={"k": 4})
                    matched_docs = db_retriever.invoke(user_question)
                    rag_context = "\n\n".join([d.page_content for d in matched_docs])
                    
                    rag_prompt = f"Dựa vào văn bản sau:\n{rag_context}\n\nHãy trả lời câu hỏi: {user_question}"
                    rag_response = client.models.generate_content(model="gemini-2.5-flash", contents=rag_prompt)
                    st.write(rag_response.text)
                except Exception as e:
                    st.error(f"Không thể xử lý câu hỏi: {e}")
