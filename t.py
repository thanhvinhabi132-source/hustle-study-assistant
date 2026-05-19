import streamlit as st
from google import genai
from PyPDF2 import PdfReader

# --- CẤU HÌNH ---
MY_API_KEY = st.secrets["GEMINI_API_KEY"]
client = genai.Client(api_key=MY_API_KEY)

# --- GIAO DIỆN ---
st.set_page_config(page_title="HUSTle Assistant", page_icon="🎓", layout="wide")

st.title("🎓 HUSTle Study Assistant")
st.markdown("*Hệ thống hỗ trợ học tập thông minh. Tải PDF để trải nghiệm sự khác biệt!*")

uploaded_file = st.file_uploader("Chọn file PDF bài giảng", type="pdf")

if uploaded_file is not None:
    with st.status("Đang xử lý dữ liệu...", expanded=True) as status:
        reader = PdfReader(uploaded_file)
        full_text = "".join([page.extract_text() for page in reader.pages])
        context_text = full_text[:15000] 
        status.update(label="Xử lý xong PDF!", state="complete", expanded=False)

    if st.button("🚀 Bắt đầu phân tích với AI", use_container_width=True):
        with st.spinner('Gemini đang thiết kế bài giảng riêng cho bạn...'):
            try:
                # BƯỚC 1: SỬA PROMPT ĐỂ AI TRẢ VỀ ĐỊNH DẠNG DỄ TÁCH
                prompt_content = f"""
                Bạn là một giáo sư tại Đại học Bách Khoa Hà Nội. Dựa trên tài liệu:
                {context_text}
                
                Hãy thực hiện:
                1. Tóm tắt kiến thức cốt lõi (bullet points).
                2. Giải thích 3 thuật ngữ khó.
                3. Tạo 5 câu hỏi trắc nghiệm. 
                
                ĐỊNH DẠNG BẮT BUỘC CHO CÂU HỎI:
                Mỗi câu hỏi viết trên 1 dòng duy nhất theo cấu trúc:
                CAU_HOI: [Nội dung câu hỏi] | A: [Đáp án A] | B: [Đáp án B] | C: [Đáp án C] | D: [Đáp án D] | DAP_AN: [Chữ cái đáp án đúng] | GIAI_THICH: [Lý do]
                """

                response = client.models.generate_content(
                    model="gemini-2.0-flash", 
                    contents=prompt_content
                )
                
                full_response = response.text
                
                # BƯỚC 2: TÁCH PHẦN TÓM TẮT VÀ PHẦN CÂU HỎI
                st.success("Phân tích hoàn tất!")
                
                # Tách phần tóm tắt (phần nằm trước câu hỏi đầu tiên)
                if "CAU_HOI:" in full_response:
                    summary_part = full_response.split("CAU_HOI:")[0]
                    question_part = full_response[full_response.find("CAU_HOI:"):]
                else:
                    summary_part = full_response
                    question_part = ""

                # Hiển thị Tóm tắt & Thuật ngữ
                st.markdown("### 📝 Tóm tắt & Thuật ngữ cốt lõi")
                st.info(summary_part)

                # Hiển thị Câu hỏi trắc nghiệm bằng TABS
                if question_part:
                    st.divider()
                    st.markdown("### 🏆 Thử thách trắc nghiệm (Chọn Tab để làm bài)")
                    
                    # Tách từng dòng câu hỏi
                    list_questions = [q for q in question_part.split("\n") if "CAU_HOI:" in q]
                    
                    # Tạo Tabs
                    tabs = st.tabs([f"Câu {i+1}" for i in range(len(list_questions))])
                    
                    for i, q_raw in enumerate(list_questions):
                        with tabs[i]:
                            try:
                                # Tách các thành phần của câu hỏi bằng dấu gạch đứng "|"
                                parts = q_raw.split("|")
                                q_text = parts[0].replace("CAU_HOI:", "").strip()
                                options = [parts[1].strip(), parts[2].strip(), parts[3].strip(), parts[4].strip()]
                                correct_ans = parts[5].replace("DAP_AN:", "").strip()
                                explain = parts[6].replace("GIAI_THICH:", "").strip()

                                # Hiển thị câu hỏi
                                st.markdown(f"**{q_text}**")
                                user_choice = st.radio(f"Chọn đáp án cho câu {i+1}:", options, index=None, key=f"radio_{i}")

                                if user_choice:
                                    # Kiểm tra xem đáp án người dùng chọn có chứa ký tự đúng không (A, B, C hoặc D)
                                    if user_choice.startswith(correct_ans):
                                        st.success(f"✅ Chính xác! Đáp án là {correct_ans}.")
                                    else:
                                        st.error(f"❌ Sai rồi! Đáp án đúng là {correct_ans}.")
                                    st.write(f"💡 **Giải thích:** {explain}")
                            except:
                                st.write("Lỗi hiển thị câu hỏi này, hãy thử bấm phân tích lại.")

            except Exception as e:
                st.error(f"Có lỗi xảy ra: {e}")
