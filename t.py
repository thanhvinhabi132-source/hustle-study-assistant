import streamlit as st  # type: ignore
from google import genai
from PyPDF2 import PdfReader
import json

# --- CẤU HÌNH SONG MÃ TRÍ TUỆ NHÂN TẠO (2 API KEYS ĐỘC LẬP) ---
# Lấy Key số 1 cho việc phân tích PDF
if "KEY_ANALYZE_PDF" in st.secrets:
    pdf_key = st.secrets["KEY_ANALYZE_PDF"]
else:
    pdf_key = None

# Lấy Key số 2 cho khung Chatbot Sidebar
if "KEY_CHATBOT_SIDEBAR" in st.secrets:
    chat_key = st.secrets["KEY_CHATBOT_SIDEBAR"]
else:
    chat_key = None

# Khởi tạo bộ não số 1 chuyên đọc tài liệu PDF
client_pdf = genai.Client(api_key=pdf_key)

# Khởi tạo trạng thái lưu dữ liệu kết quả phân tích PDF
if "ai_data" not in st.session_state:
    st.session_state["ai_data"] = None

# --- GIAO DIỆN ỨNG DỤNG CHÍNH (STREAMLIT) ---
st.set_page_config(page_title="HUSTle Assistant", page_icon="🎓", layout="centered")

st.title("🎓 HUSTle Study Assistant")
st.markdown("""
    *Hệ thống hỗ trợ học tập thông minh dành cho sinh viên Bách Khoa.*
    *Tải file PDF bài giảng lên để AI tóm tắt và đặt câu hỏi ôn tập.*
""")

with st.form("uploader_form"):
    uploaded_file = st.file_uploader("Chọn file PDF bài giảng (Slide, giáo trình...)", type="pdf")
    submit_button = st.form_submit_button("🚀 Bắt đầu phân tích với AI")

context_text = ""
if uploaded_file is not None:
    with st.status("Đang xử lý dữ liệu...", expanded=True) as status:
        st.write("Đang trích xuất văn bản từ PDF...")
        reader = PdfReader(uploaded_file)
        full_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text
        
        context_text = full_text[:15000].strip() 
        if not context_text:
            status.update(label="Lỗi: PDF không có văn bản!", state="error", expanded=True)
        else:
            status.update(label="Xử lý xong PDF! Sẵn sàng phân tích.", state="complete", expanded=False)

# SỬ DỤNG CLIENT_PDF ĐỂ XỬ LÝ (CHUYÊN TRÁCH FILE)
if submit_button:
    if not context_text:
        st.warning("Vui lòng kiểm tra lại file PDF trước khi phân tích!")
    elif not pdf_key:
        st.error("🔑 Thiếu KEY_ANALYZE_PDF trong mục Secrets để phân tích tài liệu!")
    else:
        with st.spinner('Đợi chút, Giáo sư AI đang phân tích bài giảng giúp bạn...'):
            try:
                prompt_content = f"""
                Bạn là một giáo sư tại Đại học Bách Khoa Hà Nội. 
                Dựa trên nội dung tài liệu sau đây:
                ---
                {context_text}
                ---
                Hãy phân tích và trả về cấu trúc dữ liệu theo đúng mẫu bằng tiếng Việt:
                {{
                  "summary": ["Ý tóm tắt 1", "Ý tóm tắt 2", "Ý tóm tắt 3"],
                  "terms": [
                     {{"term": "Thuật ngữ chuyên ngành 1", "definition": "Giải thích định nghĩa 1"}},
                     {{"term": "Thuật ngữ chuyên ngành 2", "definition": "Giải thích định nghĩa 2"}}
                  ],
                  "quiz": [
                     {{
                       "question": "Nội dung câu hỏi trắc nghiệm 1?",
                       "options": ["A. Đáp án A", "B. Đáp án B", "C. Đáp án C", "D. Đáp án D"],
                       "correct": "Điền chính xác nội dung câu đúng bao gồm cả chữ cái đầu",
                       "explain": "Lời giải thích ngắn gọn."
                     }}
                  ]
                }}
                Yêu cầu bắt buộc: Tạo đúng 5 câu hỏi trắc nghiệm trong mảng "quiz".
                """

                # Gọi bằng client chuyên trách PDF
                response = client_pdf.models.generate_content(
                    model="gemini-2.5-flash", 
                    contents=prompt_content,
                    config={"response_mime_type": "application/json"}
                )

                parsed_json = json.loads(response.text.strip())
                st.session_state["ai_data"] = parsed_json
                st.success("Phân tích hoàn tất!")

            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "503" in error_msg:
                    st.error("⏳ Hệ thống đọc PDF đang bận. Bạn vui lòng đợi 30 giây rồi thử lại nhé!")
                else:
                    st.error(f"Có lỗi xảy ra khi phân tích: {e}")

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
    quiz_list = data.get("quiz", []):
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


# --- CHATBOT TRỢ LÝ HUST TRÊN THANH SIDEBAR (SỬ DỤNG KEY RIÊNG BIỆT) ---
st.sidebar.markdown("## 🤖 HUST Assistant")
st.sidebar.caption("⚡ Trợ lý ảo hỗ trợ học tập Bách Khoa")

# Kiểm tra xem có Key hệ thống cho chatbot không, nếu không thì cho nhập thủ công
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
            # Khởi tạo một client hoàn toàn biệt lập chuyên phục vụ khung chat
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
                st.session_state["sidebar_chat_history"].append({"role": "assistant", "content": "⏳ Máy chủ Chatbot hiện tại đang nhận quá nhiều câu hỏi cùng lúc. Bạn đợi vài giây rồi nhấn gửi lại nhé!"})
            else:
                st.session_state["sidebar_chat_history"].append({"role": "assistant", "content": f"Lỗi kết nối: {e}"})
        
        st.rerun()
else:
    st.sidebar.warning("🔑 Vui lòng cấu hình KEY_CHATBOT_SIDEBAR trong mục Secrets để kích hoạt trợ lý ảo!")
