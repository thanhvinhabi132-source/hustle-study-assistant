import streamlit as st  # type: ignore
from google import genai
from PyPDF2 import PdfReader
import json

# --- CẤU HÌNH ---
# Gọi API Key một cách an toàn từ Secrets của Streamlit
MY_API_KEY = st.secrets["GEMINI_API_KEY"]

# Khởi tạo Client cho Gemini
client = genai.Client(api_key=MY_API_KEY)

# --- GIAO DIỆN ỨNG DỤNG (STREAMLIT) ---
st.set_page_config(page_title="HUSTle Assistant", page_icon="🎓", layout="centered")

st.title("🎓 HUSTle Study Assistant")
st.markdown("""
    *Hệ thống hỗ trợ học tập thông minh dành cho sinh viên Bách Khoa.*
    *Tải file PDF bài giảng lên để AI tóm tắt và đặt câu hỏi ôn tập.*
""")

# 1. Khu vực Upload file
uploaded_file = st.file_uploader("Chọn file PDF bài giảng (Slide, giáo trình...)", type="pdf")

if uploaded_file is not None:
    with st.status("Đang xử lý dữ liệu...", expanded=True) as status:
        st.write("Đang trích xuất văn bản từ PDF...")
        reader = PdfReader(uploaded_file)
        full_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text
        
        # Giới hạn độ dài để tránh lỗi quá tải (Lite version)
        context_text = full_text[:15000].strip() 
        
        if not context_text:
            status.update(label="Lỗi: PDF không có dữ liệu văn bản (có thể là file ảnh quét)!", state="error", expanded=True)
            st.stop()
        else:
            status.update(label="Xử lý xong PDF!", state="complete", expanded=False)

    # 2. Nút bấm phân tích
    if st.button("🚀 Bắt đầu phân tích với AI"):
        with st.spinner('Đợi chút, Gemini đang "đọc" bài và soạn đề trắc nghiệm giúp bạn...'):
            try:
                # Kỹ thuật ép Prompt trả về định dạng JSON cấu trúc
                prompt_content = f"""
                Bạn là một giáo sư tại Đại học Bách Khoa Hà Nội. 
                Dựa trên nội dung tài liệu sau đây:
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

                # Gọi API Gemini (Sử dụng model 2.5-flash mới nhất và ổn định nhất)
                response = client.models.generate_content(
                    model="gemini-2.5-flash", 
                    contents=prompt_content
                )

                # Làm sạch chuỗi phản hồi phòng trường hợp AI tự bọc tag ```json
                clean_text = response.text.strip()
                if clean_text.startswith("```json"):
                    clean_text = clean_text[7:]
                if clean_text.endswith("```"):
                    clean_text = clean_text[:-3]
                clean_text = clean_text.strip()

                # Kiểm tra và parse chuỗi JSON từ AI, sau đó lưu vào bộ nhớ Session State
                parsed_json = json.loads(clean_text)
                st.session_state["ai_data"] = parsed_json
                st.success("Phân tích hoàn tất!")

            except json.JSONDecodeError:
                st.error("Lỗi: AI phản hồi định dạng dữ liệu không chuẩn. Vui lòng bấm thử lại!")
            except Exception as e:
                st.error(f"Có lỗi xảy ra: {e}")

    # 3. Khu vực hiển thị kết quả trực quan (Giữ lại giao diện kể cả khi trang reload)
    if "ai_data" in st.session_state:
        data = st.session_state["ai_data"]
        st.divider()
        
        # Phần 1: Tóm tắt kiến thức cốt lõi
        st.markdown("### 📝 1. Tóm tắt kiến thức cốt lõi")
        for point in data.get("summary", []):
            st.markdown(f"- {point}")
            
        st.write("") # Dòng trống cho thoáng
        
        # Phần 2: Giải thích thuật ngữ chuyên ngành (Chia 3 cột)
        st.markdown("### 🔍 2. Thuật ngữ chuyên ngành cần lưu ý")
        terms_list = data.get("terms", [])
        if terms_list:
            cols = st.columns(len(terms_list))
            for idx, t in enumerate(terms_list):
                with cols[idx]:
                    st.info(f"**{t.get('term')}**\n\n*{t.get('definition')}*")

        st.write("") 

        # Phần 3: Giao diện trắc nghiệm Tab tương tác sinh động
        st.markdown("### 🧠 3. Thử thách trắc nghiệm ôn tập")
        st.caption("Hãy chọn đáp án của bạn cho từng câu hỏi dưới đây để kiểm tra kiến thức:")
        
        quiz_list = data.get("quiz", [])
        if quiz_list:
            # Tạo các tab tiêu đề câu hỏi: "Câu 1", "Câu 2",...
            tabs = st.tabs([f"Câu {x+1}" for x in range(len(quiz_list))])
            
            for i, tab in enumerate(tabs):
                with tab:
                    item = quiz_list[i]
                    st.markdown(f"**Câu hỏi:** {item.get('question')}")
                    
                    # Widget chọn đáp án (Mặc định chưa chọn câu nào nhờ index=None)
                    user_choice = st.radio(
                        "Chọn một đáp án đúng:",
                        options=item.get("options", []),
                        index=None,
                        key=f"hust_quiz_q_{i}" # Key duy nhất để không bị lẫn giữa các câu
                    )
                    
                    # Xử lý ngay khi sinh viên bấm chọn đáp án
                    if user_choice:
                        if user_choice == item.get("correct"):
                            st.success("🎉 Xuất sắc! Bạn đã trả lời đúng.")
                        else:
                            st.error(f"❌ Chưa chính xác! Đáp án đúng là: **{item.get('correct')}**")
                        
                        # Hộp thoại giải thích chi tiết ấn hiện gọn gàng
                        with st.expander("💡 Xem giải thích chi tiết từ Giáo sư"):
                            st.write(item.get("explain"))
