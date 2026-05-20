import streamlit as st  # type: ignore
from google import genai
from PyPDF2 import PdfReader
import json

# Thêm các thư viện bổ sung cho hệ thống RAG và Google GenAI chính thức
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# --- CẤU HÌNH ---
# Gọi API Key một cách an toàn từ Secrets của Streamlit
MY_API_KEY = st.secrets["GEMINI_API_KEY"]

# Khởi tạo Client cho Gemini (Dùng cho generate text thông thường)
client = genai.Client(api_key=MY_API_KEY)

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
    # Biến trạng thái kiểm tra xem hệ thống đã nạp xong cơ sở dữ liệu chưa
    if "vector_db" not in st.session_state:
        with st.status("Đang xây dựng cơ sở dữ liệu RAG...", expanded=True) as status:
            st.write("Đang trích xuất văn bản từ PDF...")
            reader = PdfReader(uploaded_file)
            full_text = ""
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
            
            # Khắc phục lỗi mã hóa utf-8 chứa ký tự surrogate từ tài liệu gốc
            full_text = full_text.encode('utf-8', 'ignore').decode('utf-8')
            
            if not full_text.strip():
                status.update(label="Lỗi: PDF không có dữ liệu văn bản (có thể là file ảnh quét)!", state="error", expanded=True)
                st.stop()
                
            st.write("Đang cắt nhỏ tài liệu thành các phân đoạn (Chunks)...")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_text(full_text)
            docs = [Document(page_content=chunk) for chunk in chunks]
            
            st.write("Đang mã hóa Vector (Embedding) bằng mô hình Google...")
            try:
                # 🔥 SỬA TÊN MÔ HÌNH THÀNH ĐỊNH DẠNG ỔN ĐỊNH NHẤT ĐỂ TRÁNH LỖI 404
                embeddings_model = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001", 
                    google_api_key=MY_API_KEY
                )
                
                st.write("Đang khởi tạo cơ sở dữ liệu tìm kiếm FAISS...")
                vector_db = FAISS.from_documents(docs, embeddings_model)
                
                st.session_state["vector_db"] = vector_db
                st.session_state["full_text_backup"] = full_text
                status.update(label="Xử lý dữ liệu RAG thành công!", state="complete", expanded=False)
            except Exception as embed_err:
                status.update(label=f"Lỗi khi tạo Vector: {embed_err}", state="error", expanded=True)
                st.stop()

    # 2. Nút bấm phân tích (Tóm tắt & Tạo đề trắc nghiệm)
    if st.button("🚀 Bắt đầu phân tích với AI"):
        with st.spinner('Đợi chút, Gemini đang "lọc" các phần quan trọng nhất để soạn đề giúp bạn...'):
            try:
                retriever = st.session_state["vector_db"].as_retriever(search_kwargs={"k": 8})
                relevant_docs = retriever.invoke("khái niệm định nghĩa lý thuyết trọng tâm công thức bài tập")
                context_text = "\n---\n".join([doc.page_content for doc in relevant_docs])

                prompt_content = f"""
                Bạn là một giáo sư tại Đại học Bách Khoa Hà Nội. 
                Dựa trên nội dung tài liệu cốt lõi sau đây:
                ---
                {context_text}
                ---
                Hãy phân tích và trả về một chuỗi JSON duy nhất (không bọc trong tag ```json hay bất kỳ ký tự nào khác) theo đúng cấu trúc mẫu sau đây bằng tiếng Việt:
                {{
                  "summary": ["Ý tóm tắt 1", "Ý tóm tắt 2", "Ý tóm tắt 3"],
                  "terms": [
                     {{"term": "Thuật ngữ chuyên ngành 1", "definition": "Giải thích định nghĩa 1"}},
                     {{"term": "Thuật ngữ chuyên ngành 2", "definition": "Giải thích định nghĩa 2"}},
                     {{"term": "Thuật ngữ chuyên ngành 3", "definition": "Giải thích định nghĩa 3"}}
                  ],
                  "quiz": [
                     {{
                       "question": "Nội dung câu hỏi trắc nghiệm 1?",
                       "options": ["A. Đáp án A", "B. Đáp án B", "C. Đáp án C", "D. Đáp án D"],
                       "correct": "Điền chính xác nội dung câu đúng bao gồm cả chữ cái đầu (Ví dụ: A. Đáp án A)",
                       "explain": "Lời giải thích tại sao câu này đúng ngắn gọn."
                     }}
                  ]
                }}
                Yêu cầu bắt buộc: Tạo đúng 5 câu hỏi trắc nghiệm trong mảng "quiz".
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

            except json.JSONDecodeError:
                st.error("Lỗi: AI phản hồi định dạng dữ liệu không chuẩn. Vui lòng bấm thử lại!")
            except Exception as e:
                st.error(f"Có lỗi xảy ra trong quá trình phân tích: {e}")

    # 3. Khu vực hiển thị kết quả trực quan
    if "ai_data" in st.session_state:
        data = st.session_state["ai_data"]
        st.divider()
        
        # Phần 1: Tóm tắt kiến thức cốt lõi
        st.markdown("### 📝 1. Tóm tắt kiến thức cốt lõi")
        for point in data.get("summary", []):
            st.markdown(f"- {point}")
            
        st.write("") 
        
        # Phần 2: Giải thích thuật ngữ chuyên ngành
        st.markdown("### 🔍 2. Thuật ngữ chuyên ngành cần lưu ý")
        terms_list = data.get("terms", [])
        if terms_list:
            num_cols = min(len(terms_list), 3)
            cols = st.columns(num_cols)
            for idx, t in enumerate(terms_list):
                col_idx = idx % num_cols
                with cols[col_idx]:
                    st.info(f"**{t.get('term')}**\n\n*{t.get('definition')}*")

        st.write("") 

        # Phần 3: Giao diện trắc nghiệm
        st.markdown("### 🧠 3. Thử thách trắc nghiệm ôn tập")
        st.caption("Hãy chọn đáp án của bạn cho từng câu hỏi dưới đây để kiểm tra kiến thức:")
        
        quiz_list = data.get("quiz", [])
        if quiz_list:
            tabs = st.tabs([f"Câu {x+1}" for x in range(len(quiz_list))])
            for i, tab in enumerate(tabs):
                with tab:
                    item = quiz_list[i]
                    st.markdown(f"**Câu hỏi:** {item.get('question')}")
                    
                    user_choice = st.radio(
