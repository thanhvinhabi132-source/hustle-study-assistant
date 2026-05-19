import streamlit as st  # type: ignore
from google import genai
from PyPDF2 import PdfReader
import json

# --- CẤU HÌNH ---
# API Key của bạn vẫn được giữ nguyên vẹn ở đây, gọi từ Streamlit Secrets
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
        
        # TĂNG TỐC 1: Giảm lượng chữ đầu vào xuống còn 8000 ký tự (vừa đủ cho 1 bài giảng)
        # Giúp Gemini đọc nhanh hơn, không bị quá tải
        context_text = full_text[:8000].strip() 
        
        if not context_text:
            status.update(label="Lỗi: PDF không có dữ liệu văn bản!", state="error", expanded=True)
            st.stop()
        else:
            status.update(label="Xử lý xong PDF!", state="complete", expanded=False)

    # 2. Nút bấm phân tích
    if st.button("🚀 Bắt đầu phân tích với AI"):
        with st.spinner('Đợi chút, Gemini đang xử lý siêu tốc giúp bạn...'):
            try:
                # Prompt yêu cầu cấu trúc rõ ràng
                prompt_content = f"""
                Bạn là một giáo sư tại Đại học Bách Khoa Hà Nội. 
                Dựa trên nội dung tài liệu sau đây:
                ---
                {context_text}
                ---
                Hãy phân tích và trả về một chuỗi JSON theo đúng cấu trúc mẫu sau bằng tiếng Việt:
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
                       "correct": "Nội dung câu đúng gồm cả chữ cái đầu (Ví dụ: A. Đáp án A)",
                       "explain": "Lời giải thích ngắn gọn."
                     }}
                  ]
                }}
                Yêu cầu: Tạo đúng 5 câu hỏi trắc nghiệm trong mảng "quiz".
                """

                # TĂNG TỐC 2: Cấu hình API để tối ưu tốc độ phản hồi
                response = client.models.generate_content(
                    model="gemini-2.5-flash", 
                    contents=prompt_content,
                    config={
                        # Ép trả về JSON bằng tính năng hệ thống (Nhanh và chính xác nhất)
                        "response_mime_type": "application/json",
                        # Giới hạn số ký tự trả về để AI viết ngắn gọn, tập trung vào cấu trúc
                        "max_output_tokens": 1200, 
                        # Giảm độ sáng tạo để AI phản hồi nhanh và ít lỗi logic hơn
                        "temperature": 0.2, 
                    }
                )

                # Kiểm tra và parse chuỗi JSON từ AI
                parsed_json = json.loads(response.text.strip())
                st.session_state["ai_data"] = parsed_json
                st.success("Phân tích hoàn tất!")

            except json.JSONDecodeError:
                st.error("Lỗi hệ thống: Định dạng dữ liệu không đồng nhất. Vui lòng thử lại!")
            except Exception as e:
                st.error(f"Có lỗi xảy ra: {e}")

    # 3. Khu vực hiển thị kết quả trực quan
    if "ai_data" in st.session_state:
        data = st.session_state["ai_data"]
        st.divider()
        
        # Phần 1: Tóm tắt kiến thức
        st.markdown("### 📝 1. Tóm tắt kiến thức cốt lõi")
        for point in data.get("summary", []):
            st.markdown(f"- {point}")
            
        st.write("") 
        
        # Phần 2: Giải thích thuật ngữ
        st.markdown("### 🔍 2. Thuật ngữ chuyên ngành cần lưu ý")
        terms_list = data.get("terms", [])
        if terms_list:
            cols = st.columns(len(terms_list))
            for idx, t in enumerate(terms_list):
                with cols[idx]:
                    st.info(f"**{t.get('term')}**\n\n*{t.get('definition')}*")

        st.write("") 

        # Phần 3: Giao diện trắc nghiệm Tab tương tác
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
                        "Chọn một đáp án đúng:",
                        options=item.get("options", []),
                        index=None,
                        key=f"hust_quiz_q_{i}" 
                    )
                    
                    if user_choice:
                        if user_choice == item.get("correct"):
                            st.success("🎉 Xuất sắc! Bạn đã trả lời đúng.")
                        else:
                            st.error(f"❌ Chưa chính xác! Đáp án đúng là: **{item.get('correct')}**")
                        
                        with st.expander("💡 Xem giải thích chi tiết từ Giáo sư"):
                            st.write(item.get("explain"))
