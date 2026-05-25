import streamlit as st  # type: ignore
from google import genai
from google.genai import types
import json

# --- CẤU HÌNH SONG MÃ TRÍ TUỆ NHÂN TẠO (2 API KEYS ĐỘC LẬP) ---
if "KEY_ANALYZE_PDF" in st.secrets:
    pdf_key = st.secrets["KEY_ANALYZE_PDF"]
else:
    pdf_key = None

if "KEY_CHATBOT_SIDEBAR" in st.secrets:
    chat_key = st.secrets["KEY_CHATBOT_SIDEBAR"]
else:
    chat_key = None

client_pdf = genai.Client(api_key=pdf_key)

if "ai_data" not in st.session_state:
    st.session_state["ai_data"] = None

# --- GIAO DIỆN ỨNG DỤNG CHÍNH ---
st.set_page_config(page_title="HUSTle Assistant", page_icon="🎓", layout="centered")

st.title("🎓 HUSTle Study Assistant")
st.markdown("""
    *Hệ thống hỗ trợ học tập thông minh dành cho sinh viên Bách Khoa.*
    *Tải file PDF (Đọc được cả văn bản số và ảnh quét/scan) để AI tóm tắt và đặt câu hỏi.*
""")

with st.form("uploader_form"):
    uploaded_file = st.file_uploader("Chọn file PDF bài giảng (Slide, giáo trình, ảnh scan...)", type="pdf")
    submit_button = st.form_submit_button("🚀 Bắt đầu phân tích với AI")

# XỬ LÝ GỬI FILE THÔ SANG GEMINI
if submit_button:
    if uploaded_file is None:
        st.warning("Vui lòng tải file PDF lên trước!")
    elif not pdf_key:
        st.error("🔑 Thiếu KEY_ANALYZE_PDF trong mục Secrets để phân tích tài liệu!")
    else:
        with st.spinner('Đợi chút, Giáo sư AI đang "nhìn" và phân tích toàn bộ trang PDF (kể cả hình ảnh) giúp bạn...'):
            try:
                # Đọc file PDF thành dữ liệu bytes thô
                pdf_bytes = uploaded_file.read()

                # Cấu hình Prompt yêu cầu AI đọc và phân tích dữ liệu đa phương thức
                prompt_content = """
                Bạn là một giáo sư tại Đại học Bách Khoa Hà Nội. 
                Hãy đọc và phân tích kỹ tài liệu PDF được đính kèm (bao gồm cả việc nhìn các hình ảnh, chữ quét scan trong file nếu có).
                
                Sau đó, hãy lập tức trả về cấu trúc dữ liệu theo đúng mẫu bằng tiếng Việt:
                {
                  "summary": ["Ý tóm tắt quan trọng 1", "Ý tóm tắt quan trọng 2", "Ý tóm tắt quan trọng 3"],
                  "terms": [
                     {"term": "Thuật ngữ chuyên ngành 1", "definition": "Giải thích định nghĩa 1"},
                     {"term": "Thuật ngữ chuyên ngành 2", "definition": "Giải thích định nghĩa 2"}
                  ],
                  "quiz": [
                     {
                       "question": "Nội dung câu hỏi trắc nghiệm 1?",
                       "options": ["A. Đáp án A", "B. Đáp án B", "C. Đáp án C", "D. Đáp án D"],
                       "correct": "Điền chính xác nội dung câu đúng bao gồm cả chữ cái đầu (Ví dụ: A. Đáp án A)",
                       "explain": "Lời giải thích ngắn gọn tại sao đúng."
                     }
                  ]
                }
                Yêu cầu bắt buộc: Tạo đúng 5 câu hỏi trắc nghiệm trong mảng "quiz".
                """

                # Sử dụng types.Part.from_bytes để truyền thẳng file PDF sang cho Gemini tự OCR
                response = client_pdf.models.generate_content(
                    model="gemini-2.5-flash", 
                    contents=[
                        types.Part.from_bytes(
                            data=pdf_bytes,
                            mime_type="application/pdf",
                        ),
                        prompt_content
                    ],
                    config={"response_mime_type": "application/json"}
                )

                parsed_json = json.loads(response.text.strip())
                st.session_state["ai_data"] = parsed_json
                st.success("Phân tích hoàn tất!")

            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "503" in error_msg:
                    st.error("⏳ Hệ thống đọc PDF đang bận do kích thước file hoặc quá tải. Bạn vui lòng đợi một chút rồi thử lại nhé!")
                else:
                    st.error(f"Có lỗi xảy ra khi phân tích: {e}")

# HIỂN THỊ KẾT QUẢ MÀN HÌNH CHÍNH
if st.session_state["ai_data"] is not None:
    data = st.session_state["ai_data"]
    st.divider()
    
    st.markdown("### 📝 1. Tóm tắt kiến thức cốt lõi")
    for point in data.get("summary", []):
        st.markdown(f"- {point}")
        
    st.write("") 
    
    st.markdown("### 🔍 2. Thuật ngữ chuyên ngành cần lưu ý")
    terms_list = data.get("terms", [])
    if terms_list:
        cols = st.columns(len(terms_list))
        for idx, t in enumerate(terms_list):
            with cols[idx]:
                st.info(f"**{t.get('term')}**\n\n*{t.get('definition')}*")

    st.write("") 

    st.markdown("### 🧠 3. Thử thách trắc nghiệm ôn tập")
    st.caption("Hãy chọn đáp án của bạn cho từng câu hỏi dưới đây để kiểm tra kiến thức:")
    
    quiz_list = data.get("quiz", [])
    if quiz_list:
        tabs = st.tabs([f"Câu {x+1}" for x in range(len(quiz_list))])
        for i, tab in enumerate(tabs):
            with tab:
                item = quiz_list[i]
                st.markdown(f"**Câu hỏi:** {item.get('question')}")
                user_choice = st.radio("Chọn một đáp án đúng:", options=item.get("options", []), index=None, key=f"hust_quiz_q_{i}")
                if user_choice:
                    if user_choice == item.get("correct"):
                        st.success("🎉 Xuất sắc!")
                    else:
                        st.error(f"❌ Chưa chính xác! Đáp án đúng là: **{item.get('correct')}**")
                    with st.expander("💡 Xem giải thích chi tiết từ Giáo sư"):
                        st.write(item.get("explain"))


# --- CHATBOT TRỢ LÝ HUST TRÊN THANH SIDEBAR ---
st.sidebar.markdown("## 🤖 HUST Assistant")
st.sidebar.caption("⚡ Trợ lý ảo hỗ trợ học tập Bách Khoa")

if chat_key:
    api_key_to_use = chat_key
else:
    api_key_to_use = st.sidebar.text_input("Nhập Gemini API Key dự phòng để chat:", type="password")

st.sidebar.divider()

if api_key_to_use:
    if "sidebar_chat_history" not in st.session_state:
        st.session_state["sidebar_chat_history"] = [
            {"role": "assistant", "content": "Xin chào! Tôi là HUST Assistant. Tôi chạy bằng luồng API Key độc lập nên tốc độ phản hồi sẽ cực kỳ nhanh và không lo nghẽn mạng nhé!"}
        ]

    with st.sidebar.container():
        for message in st.session_state["sidebar_chat_history"]:
            with st.sidebar.chat_message(message["role"]):
                st.sidebar.markdown(message["content"])

    user_query = st.sidebar.chat_input("Hỏi trợ lý HUST...", key="sidebar_chat_input")

    if user_query:
        st.session_state["sidebar_chat_history"].append({"role": "user", "content": user_query})
        
        try:
            client_chat = genai.Client(api_key=api_key_to_use)
            system_instruction = (
                "Bạn là HUST Assistant - trợ lý ảo thông minh của Đại học Bách Khoa Hà Nội. "
                "Hãy trả lời ngắn gọn, tập trung thẳng vào câu hỏi bằng tiếng Việt."
            )
            
            response = client_chat.models.generate_content(
                model="gemini-2.5-flash",
                contents=user_query,
                config={"system_instruction": system_instruction}
            )
            
            ai_reply = response.text if response.text else "Tôi chưa rõ ý bạn."
            st.session_state["sidebar_chat_history"].append({"role": "assistant", "content": ai_reply})
            
        except Exception as e:
            error_msg = str(e)
            if "503" in error_msg or "UNAVAILABLE" in error_msg:
                st.sidebar.error("⏳ Máy chủ bận, thử lại sau vài giây.")
            else:
                st.sidebar.error(f"Lỗi: {e}")
        st.rerun()
