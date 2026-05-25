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

# GIẢI PHÁP 1: Bọc khu vực nhập liệu vào st.form để cô lập hành động click, tránh re-run gọi API trùng lặp
with st.form("uploader_form"):
    uploaded_file = st.file_uploader("Chọn file PDF bài giảng (Slide, giáo trình...)", type="pdf")
    submit_button = st.form_submit_button("🚀 Bắt đầu phân tích với AI")

# Xử lý đọc file PDF (Luôn chạy độc lập với nút bấm AI để tối ưu trải nghiệm)
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
        
        # Giới hạn độ dài văn bản để tối ưu số lượng Token đầu vào
        context_text = full_text[:15000].strip() 
        
        if not context_text:
            status.update(label="Lỗi: PDF không có dữ liệu văn bản (có thể là file ảnh quét)!", state="error", expanded=True)
        else:
            status.update(label="Xử lý xong PDF! Sẵn sàng phân tích.", state="complete", expanded=False)

# Kích hoạt khi người dùng nhấn nút Submit trong Form
if submit_button:
    if not context_text:
        st.warning("Vui lòng kiểm tra lại file PDF trước khi phân tích!")
    else:
        with st.spinner('Đợi chút, Gemini đang "đọc" bài và soạn đề trắc nghiệm giúp bạn...'):
            try:
                # Cấu trúc Prompt tinh gọn hơn vì đã có cấu hình JSON hệ thống xử lý định dạng
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
                       "correct": "Điền chính xác nội dung câu đúng bao gồm cả chữ cái đầu (Ví dụ: A. Đáp án A)",
                       "explain": "Lời giải thích tại sao câu này đúng ngắn gọn."
                     }}
                  ]
                }}
                Yêu cầu bắt buộc: Tạo đúng 5 câu hỏi trắc nghiệm trong mảng "quiz".
                """

                # GIẢI PHÁP 2: Sử dụng `response_mime_type` để ép Gemini trả ra JSON thuần, loại bỏ lỗi cú pháp văn bản
                response = client.models.generate_content(
                    model="gemini-2.5-flash", 
                    contents=prompt_content,
                    config={
                        "response_mime_type": "application/json"
                    }
                )

                # Chuyển đổi dữ liệu chuỗi JSON từ AI thành Dictionary Python và lưu vào Session State
                parsed_json = json.loads(response.text.strip())
                st.session_state["ai_data"] = parsed_json
                st.success("Phân tích hoàn tất!")

            # GIẢI PHÁP 3: Xử lý và bắt lỗi quá tải (Rate Limit 429) một cách thân thiện
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg:
                    st.error("⏳ Hệ thống đang bị quá tải (Rate Limit) do có quá nhiều yêu cầu cùng lúc. Bạn vui lòng đợi khoảng 1 phút rồi bấm nút thử lại nhé!")
                elif "JSONDecodeError" in error_msg or "json" in error_msg.lower():
                    st.error("❌ Lỗi cấu trúc dữ liệu từ AI. Hãy thử bấm lại nút phân tích một lần nữa!")
                else:
                    st.error(f"Có lỗi hệ thống xảy ra: {e}")

# --- KHIU VỰC HIỂN THỊ KẾT QUẢ (Giữ nguyên giao diện mượt mà của bạn) ---
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
                    
                    with st.expand_visible if hasattr(st, "expand_visible") else st.expander("💡 Xem giải thích chi tiết từ Giáo sư"):
                        st.write(item.get("explain"))




# --- CHATBOT PHONG CÁCH MESSENGER CHUẨN CẤU TRÚC (ĐẶT SÁT LỀ TRÁI) ---

# Khởi tạo trạng thái đóng/mở của khung chat
if "chat_open" not in st.session_state:
    st.session_state["chat_open"] = False

# 1. Nhúng CSS để ép toàn bộ cụm chat cố định hoàn toàn ở góc dưới bên phải
st.html("""
    <style>
        /* Ép nút bấm luôn nằm cố định ở góc dưới cùng bên phải màn hình */
        .stButton > button[key="messenger_toggle_btn"] {
            position: fixed !important;
            bottom: 30px !important;
            right: 30px !important;
            z-index: 999999 !important;
            background: linear-gradient(135deg, #FF4B4B 0%, #FF8533 100%) !important;
            color: white !important;
            font-weight: bold !important;
            border-radius: 50px !important;
            padding: 12px 24px !important;
            box-shadow: 0px 4px 15px rgba(255, 75, 75, 0.4) !important;
            border: none !important;
        }

        /* Khung chứa chat box cố định góc phải, nổi lên trên nền JSON */
        div.fixed-chat-window {
            position: fixed !important;
            bottom: 95px !important;
            right: 30px !important;
            width: 380px !important;
            background-color: #1E1E1E !important; /* Đổi sang nền tối để hợp với style Bách Khoa và nổi bật viền */
            border-radius: 16px !important;
            box-shadow: 0px 8px 32px rgba(0,0,0,0.5) !important;
            z-index: 999998 !important;
            border: 2px solid #FF4B4B !important;
            padding: 15px !important;
        }
        
        /* Làm đẹp phần chữ tiêu đề trong khung chat */
        div.fixed-chat-window h4 {
            color: #FFFFFF !important;
            margin-bottom: 2px !important;
        }
    </style>
""")

# 2. Hiển thị nút bấm bong bóng chat
if st.button("💬 Trợ lý HUST", key="messenger_toggle_btn"):
    st.session_state["chat_open"] = not st.session_state["chat_open"]
    st.rerun()

# 3. Nếu trạng thái mở, tạo cửa sổ Chatbox bọc trong Class CSS cố định góc màn hình
if st.session_state["chat_open"]:
    # Bắt đầu bọc khung chat
    st.markdown('<div class="fixed-chat-window">', unsafe_allow_html=True)
    
    st.markdown("#### 🤖 HUST Assistant")
    st.caption("⚡ Trợ lý ảo đang trực tuyến")
    st.markdown("<hr style='margin: 8px 0; border-color: #444;'/>", unsafe_allow_html=True)
    
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = [
            {"role": "assistant", "content": "Xin chào! Tôi là HUST Assistant. Tôi có thể giúp gì cho bạn?"}
        ]

    # Khung chứa nội dung tin nhắn (Giới hạn chiều cao để nằm gọn trong bong bóng)
    chat_container = st.container(height=250, border=False)
    with chat_container:
        for message in st.session_state["chat_history"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Ô nhập câu hỏi
    user_query = st.chat_input("Nhập câu hỏi...", key="msn_input_fixed")

    if user_query:
        st.session_state["chat_history"].append({"role": "user", "content": user_query})
        
        try:
            system_instruction = (
                "Bạn là HUST Assistant - trợ lý ảo thông minh của Đại học Bách Khoa Hà Nội. "
                "Hãy trả lời ngắn gọn, tập trung thẳng vào câu hỏi bằng tiếng Việt."
            )
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=user_query,
                config={"system_instruction": system_instruction}
            )
            ai_reply = response.text if response.text else "Tôi chưa rõ ý bạn."
            st.session_state["chat_history"].append({"role": "assistant", "content": ai_reply})
            
        except Exception as e:
            if "429" in str(e):
                st.session_state["chat_history"].append({"role": "assistant", "content": "⏳ Hệ thống bận, thử lại sau vài giây."})
            else:
                st.session_state["chat_history"].append({"role": "assistant", "content": f"Lỗi: {e}"})
        
        st.rerun()

    # Đóng bọc khung chat
    st.markdown('</div>', unsafe_allow_html=True)
