import streamlit as st # type: ignore
from google import genai
from PyPDF2 import PdfReader

# --- CẤU HÌNH ---
# 1. Dán API Key bạn vừa lấy ở Bước 2 vào đây
MY_API_KEY = "AIzaSyCTO_3RRu1wczYoGMwPdskYCeti7UWLzNI"  # <--- THAY BẰNG KEY CỦA BẠN

# 2. Khởi tạo Client cho Gemini
client = genai.Client(api_key=MY_API_KEY)

# --- GIAO DIỆN ỨNG DỤNG (STREAMLIT) ---
st.set_page_config(page_title="HUSTle Assistant", page_icon="🎓")

st.title("🎓 HUSTle Study Assistant")
st.markdown("""
    *Hệ thống hỗ trợ học tập thông minh dành cho sinh viên Bách Khoa.*
    *Tải file PDF bài giảng lên để AI tóm tắt và đặt câu hỏi ôn tập.*
""")

# 1. Khu vực Upload file
uploaded_file = st.file_uploader("Chọn file PDF bài giảng (Slide, giáo trình...)", type="pdf")

if uploaded_file is not None:
    with st.status("Đang xử lý dữ liệu...", expanded=True) as status:
        # 2. Đọc nội dung văn bản từ PDF
        st.write("Đang trích xuất văn bản từ PDF...")
        reader = PdfReader(uploaded_file)
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text()
        
        # Giới hạn độ dài để tránh lỗi quá tải (Lite version)
        context_text = full_text[:15000] 
        status.update(label="Xử lý xong PDF!", state="complete", expanded=False)

    # 3. Nút bấm phân tích
    if st.button("🚀 Bắt đầu phân tích với AI"):
        with st.spinner('Đợi chút, Gemini đang "đọc" bài giúp bạn...'):
            try:
                # Prompt kỹ thuật để tối ưu kết quả
                prompt_content = f"""
                Bạn là một giáo sư tại Đại học Bách Khoa Hà Nội. 
                Dựa trên nội dung tài liệu sau đây:
                ---
                {context_text}
                ---
                Hãy thực hiện các yêu cầu sau bằng tiếng Việt:
                1. Tóm tắt các kiến thức cốt lõi nhất dưới dạng danh sách (bullet points).
                2. Giải thích 3 thuật ngữ chuyên ngành khó nhất trong bài.
                3. Tạo 5 câu hỏi trắc nghiệm ôn tập (có đáp án giải thích bên dưới).
                """

                # Gọi API Gemini
                response = client.models.generate_content(
                    model="gemini-3-flash-preview", 
                    contents=prompt_content
                )

               # 4. Hiển thị kết quả ra màn hình
                st.success("Phân tích hoàn tất!")
                st.divider() # Tạo một đường kẻ ngang cho đẹp
                st.markdown("### 📝 Kết quả từ AI:")
                st.markdown(response.text) # Đây chính là dòng 63 đã sửa
            except Exception as e:
                st.error(f"Có lỗi xảy ra: {e}")
