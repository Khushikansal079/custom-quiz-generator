from fpdf import FPDF
import streamlit as st
from mcq_generator import AdvancedMCQGenerator
from short_answer_generator import QuestionGenerator
from truefalse_quiz import generate_true_false
import io
import fitz

# ---------------- PDF GENERATION FUNCTION ---------------- #
def generate_pdf(text: str) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)

    for line in text.split('\n'):
        pdf.multi_cell(0, 10, line)

    
    pdf_bytes = pdf.output(dest='S').encode('latin1')
    return pdf_bytes

# ---------------- STYLING ---------------- #
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(to right, #a18cd1, #fbc2eb);
            background-attachment: fixed;
        }
        .css-1d391kg, .css-18ni7ap {
            background-color: rgba(255, 255, 255, 0.8) !important;
            border-radius: 12px;
            padding: 1rem;
        }
        h1, h2, h3, h4, h5, h6, p, div, span {
            color: #2C3E50 !important;
        }
        .stButton>button {
            background-color: #F67280;
            color: white;
            border-radius: 8px;
            border: none;
            padding: 0.5rem 1rem;
        }
        .stButton>button:hover {
            background-color: #e15b6f;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------- PAGE SETUP ---------------- #
st.set_page_config(page_title="EduGenie", layout="centered")
st.title("📝 EduGenie")
st.markdown("🌟From context to quiz in seconds– EduGenie grants your learning wishes with AI precision.")

# ---------------- INPUT SECTION ---------------- #
uploaded_file = st.file_uploader("📄 Upload a PDF file", type="pdf")
pdf_text = ""

if uploaded_file is not None:
    try:
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        for page in doc:
            pdf_text += page.get_text()
        st.success("✅ PDF uploaded successfully!")
    except Exception as e:
        st.error(f"Failed to read PDF: {str(e)}")

context = pdf_text if pdf_text else st.text_area("📜 Enter your context/text here:", height=100)

col1, col2 = st.columns(2)
question_type = col1.selectbox("Question Type", ["Multiple Choice", "Short Answer", "True/False"])
difficulty = col2.selectbox("Difficulty", ["easy", "medium", "hard"])
num_questions = st.slider("🔢 Number of Questions", min_value=1, max_value=10, value=3)

# ---------------- QUIZ GENERATION ---------------- #
if st.button("⚡ Generate Quiz"):
    if not context.strip():
        st.warning("Please enter some context/text to generate questions.")
    else:
        with st.spinner("Generating quiz..."):
            output = io.StringIO()
            questions = []

            if question_type == "Multiple Choice":
                generator = AdvancedMCQGenerator()
                try:
                    questions = generator.generate_mcq(context, num_questions=num_questions, difficulty=difficulty)
                    st.subheader("📘 Multiple Choice Questions")
                    for idx, q in enumerate(questions, 1):
                        st.markdown(f"**Q{idx}: {q['question']}**")
                        for i, option in enumerate(q['options']):
                            st.markdown(f"- {chr(65+i)}. {option}")
                        st.markdown(f"🟢 **Answer:** {chr(65 + q['correct_answer'])}\n\n---")
                        output.write(f"Q{idx}: {q['question']}\n")
                        for i, option in enumerate(q['options']):
                            output.write(f"  {chr(65+i)}. {option}\n")
                        output.write(f"Answer: {chr(65 + q['correct_answer'])}\n\n")
                except Exception as e:
                    st.error(f"❌ Failed to generate MCQs: {str(e)}")

            elif question_type == "Short Answer":
                try:
                    generator = QuestionGenerator()
                    questions = generator.generate_questions(context, num_questions=num_questions, difficulty=difficulty)
                    st.subheader("📝 Short Answer Questions")
                    for idx, q in enumerate(questions, 1):
                        st.markdown(f"**Q{idx}: {q['question']}**")
                        st.markdown(f"🟢 **Expected Keyword:** {q['answer']}")
                        st.markdown("---")
                        output.write(f"Q{idx}: {q['question']}\nExpected keyword: {q['answer']}\n\n")
                except Exception as e:
                    st.error(f"❌ Failed to generate short answer questions: {str(e)}")

            elif question_type == "True/False":
                try:
                    st.subheader("✅ True/False Questions")
                    tf_generator = generate_true_false()
                    sentences = tf_generator.validate_inputs(context, num_questions, difficulty)
                    questions = tf_generator.generate_statements(context, num_questions, difficulty, sentences)

                    for idx, (statement, label) in enumerate(questions, 1):
                        st.markdown(f"**Q{idx}: {statement}**")
                        st.markdown(f"🟢 **Answer:** {'True' if label == 'ENTAILMENT' else 'False'}")
                        st.markdown("---")
                        output.write(f"Q{idx}: {statement}\nAnswer: {'True' if label == 'ENTAILMENT' else 'False'}\n\n")
                except Exception as e:
                    st.error(f"❌ Failed to generate true/false questions: {str(e)}")

            # ---------------- PDF DOWNLOAD ---------------- #
            if questions:
                pdf_bytes = generate_pdf(output.getvalue())
                st.download_button("⬇️ Download Quiz as PDF", data=pdf_bytes, file_name="EduGenie_quiz.pdf", mime="application/pdf")
