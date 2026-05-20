import streamlit as st  # type: ignore
from google import genai
from PyPDF2 import PdfReader
import json

# Các thư viện bổ sung cho cấu trúc dữ liệu RAG
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# --- CẤU HÌNH ---
# Gọi API Key một cách an toàn từ Secrets của Streamlit
MY_API_KEY = st.secrets["GEMINI_API_KEY"]

# Khởi tạo Client duy nhất cho SDK chính thức của Google GenAI
client = genai.Client(api_key=MY_API_KEY)

# Lớp tùy biến tích hợp Vector Embedding thế hệ mới sử dụng SDK chính thức
class GeminiEmbeddings:
    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            try:
                # Làm sạch dữ liệu chuỗi để tránh lỗi surrogate cặp ký tự lỗi
                safe_text = text.encode('utf-8', 'ignore').decode('utf-8')
                if not safe_text.strip():
                    safe_text = "blank"
                response = client.models.embed_content(
                    model="text-embedding-004",
                    contents=safe_text
                )
                embeddings.append(response.embeddings[0].values)
            except Exception:
                # Tạo vector giả định dạng 768 chiều nếu một đoạn văn bản nhỏ bị lỗi nặng
                embeddings.append([0.0] * 768)
        return embeddings

    def embed_query(self, text):
        safe_text = text.encode('utf-8', 'ignore').decode('utf-8')
        response = client.models.embed_content(
            model="text-embedding-004",
            contents=safe_text
        )
        return response.embeddings[0].values

    # 🔥 SỬA LỖI 'not callable': Cho phép FAISS gọi trực tiếp object này khi thực hiện truy vấn
    def __call__(self, text):
        if isinstance(text, list):
            return self.embed_documents(text)
        return self.embed_query(text)

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
            
            # Khắc phục lỗi mã hóa utf-8 chứa ký tự đặc biệt lạ
            full_text = full_text.encode('utf-8', 'ignore').decode('utf-8')
            
            if not full_text.strip():
                status.update(label="Lỗi: PDF không có dữ liệu văn bản (file ảnh quét)!", state="error", expanded=True)
                st.stop()
                
            st.write("Đang cắt nhỏ tài liệu thành các phân đoạn (Chunks)...")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_text(full_text)
            docs = [Document(page_content=chunk) for chunk in chunks]
            
            st.write("Đang mã hóa Vector (Embedding) bằng mô hình Google...")
            try:
                embeddings_model = GeminiEmbeddings()
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
                    
                    opts = item.get("options", [])
                    user_choice = st.radio("Chọn một đáp án đúng:", options=opts, index=None, key=f"q_final_fixed_{i}")
                    
                    if user_choice:
                        if user_choice == item.get("correct"):
                            st.success("🎉 Xuất sắc! Bạn đã trả lời đúng.")
                        else:
                            st.error(f"❌ Chưa chính xác! Đáp án đúng là: **{item.get('correct')}**")
                        
                        with st.expander("💡 Xem giải thích chi tiết từ Giáo sư"):
                            st.write(item.get("explain"))

        # Khung hỏi đáp Q&A ứng dụng RAG nâng cao
        st.divider()
        st.markdown("### 💬 4. Hỏi đáp Siêu tốc về Tài liệu (RAG Q&A)")
        st.caption("Bạn có thắc mắc gì thêm về bài học này không? Hãy đặt câu hỏi, AI sẽ tự động lục tìm đúng vị trí trong file để trả lời.")
        
        user_question = st.text_input("Nhập câu hỏi của bạn (Ví dụ: Định lý này áp dụng khi nào?, Công thức tính X là gì?):", key="rag_query_input")
        
        if user_question:
            with st.spinner("Đang truy vấn dữ liệu và phân tích văn bản..."):
                try:
                    db_retriever = st.session_state["vector_db"].as_retriever(search_kwargs={"k": 4})
                    matched_docs = db_retriever.invoke(user_question)
                    
                    rag_context = "\n\n".join([f"[Đoạn tham khảo {idx+1}]: {d.page_content}" for idx, d in enumerate(matched_docs)])
                    
                    rag_prompt = f"""
                    Bạn là giảng viên Đại học Bách Khoa Hà Nội, đang giải đáp thắc mắc cho sinh viên.
                    Hãy trả lời câu hỏi sau đây một cách chính xác, mạch lạc dựa trên phần tài liệu tham khảo được trích xuất từ giáo trình.

                    TÀI LIỆU THAM KHẢO CHÍNH XÁC:
                    {rag_context}

                    CÂU HỎI CỦA SINH VIÊN:
                    {user_question}

                    Yêu cầu: Trả lời ngắn gọn, tập trung thẳng vào câu hỏi, sử dụng ngôn từ sư phạm dễ hiểu. Nếu tài liệu tham khảo trên không chứa thông tin để trả lời, hãy báo rằng "Tài liệu được tải lên không có thông tin chi tiết về phần này" chứ không tự bịa ra thông tin.
                    """
                    
                    rag_response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=rag_prompt
                    )
                    
                    st.markdown("#### 👨‍🏫 Câu trả lời từ trợ lý:")
                    st.write(rag_response.text)
                    
                    with st.expander("🔍 Xem các nguồn thông tin được trích lục từ PDF"):
                        for idx, doc in enumerate(matched_docs):
                            st.info(f"**Nguồn {idx+1}:** {doc.page_content}")
                            
                except Exception as e:
                    st.error(f"Không thể xử lý câu hỏi: {e}")
