import streamlit as st  # type: ignore
from google import genai
from pydantic import BaseModel, Field
from PyPDF2 import PdfReader
import json

# --- ĐỊNH NGHĨA CẤU TRÚC DỮ LIỆU CỨNG (STRUCTURED OUTPUTS) ---
class TermItem(BaseModel):
    term: str = Field(description="Thuật ngữ chuyên ngành")
    definition: str = Field(description="Định nghĩa chi tiết của thuật ngữ")

class QuizItem(BaseModel):
    question: str = Field(description="Nội dung câu hỏi trắc nghiệm")
    options: list[str] = Field(description="Danh sách gồm đúng 4 lựa chọn, ví dụ: ['A. ...', 'B. ...', 'C. ...', 'D. ...']")
    correct: str = Field(description="Nội dung chính xác của đáp án đúng, phải khớp hoàn toàn với một phần tử trong options bao gồm cả ký tự đầu (Ví dụ: A. ...)")
    explain: str = Field(description="Lời giải thích lý do đáp án đó đúng ngắn gọn")

class StudyAnalysis(BaseModel):
    summary: list[str] = Field(description="Danh sách các ý tóm tắt kiến thức cốt lõi nhất")
    terms: list[TermItem] = Field(description="Danh sách gồm 3 thuật ngữ chuyên ngành khó nhất")
    quiz: list[QuizItem] = Field(description="Danh sách gồm đúng 5 câu hỏi trắc nghiệm ôn tập")


# --- CẤU HÌNH ---
MY_API_KEY = st.secrets["GEMINI_API_KEY"]
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
        
        # Cắt text hợp lý để tránh overload
        context_text = full_text[:10000].strip() 
        
        if not context_text:
            status.update(label="Lỗi: PDF không có dữ liệu văn bản!", state="error", expanded=True)
            st.stop()
        else:
            status.update(label="Xử lý xong PDF!", state="complete", expanded=False)

    # 2. Nút bấm phân tích
    if st.button("🚀 Bắt đầu phân tích với AI"):
        with st.spinner('Đợi chút, Gemini đang phân tích và tạo bộ trắc nghiệm chuẩn cấu trúc...'):
            try:
                prompt_content = f"""
                Bạn là một giáo sư tại Đại học Bách Khoa Hà Nội. 
                Dựa trên nội dung tài liệu sau đây:
                ---
                {context_text}
                ---
                Hãy thực hiện tóm tắt các ý chính, giải thích thuật ngữ chuyên ngành khó và tạo đúng 5 câu hỏi trắc nghiệm ôn tập dựa trên tài liệu.
                """

                # Gọi API với cấu hình response_schema nghiêm ngặt
                response = client.models.generate_content(
                    model="gemini-2.5-flash", 
                    contents=prompt_content,
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": StudyAnalysis,  # Ép cấu hình schema Pydantic cứng
                        "max_output_tokens": 2000,
                        "temperature": 0.1, 
                    }
                )

                # Parse dữ liệu an toàn từ kết quả JSON chắc chắn chuẩn định dạng
                parsed_json = json.loads(response.text.strip())
                st.session_state["ai_data"] = parsed_json
                st.success("Phân tích hoàn tất!")

            except Exception as e:
                st.error(f"Có lỗi xảy ra trong quá trình xử lý: {e}")

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
