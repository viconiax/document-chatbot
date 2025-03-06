import streamlit as st
import openai
import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.settings import Settings

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io

def generate_pdf(content, filename="generated_document.pdf"):
    """Generate a downloadable PDF from given text content."""
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    pdf.setFont("Helvetica", 12)

    # Split the text into lines and write them to the PDF
    y_position = 750  # Start writing from the top
    for line in content.split("\n"):
        pdf.drawString(50, y_position, line)
        y_position -= 20  # Move down for next line

        # Ensure we don't write off the page
        if y_position < 50:
            pdf.showPage()
            pdf.setFont("Helvetica", 12)
            y_position = 750

    pdf.save()
    buffer.seek(0)

    return buffer


# 🔹 Load OpenAI API Key
openai_api_key = st.secrets["OPENAI_API_KEY"]
openai_client = openai.OpenAI(api_key=openai_api_key)

# 🔹 Streamlit App UI
st.title("📄 AI Job Description & Case Study Generator")
st.write("This chatbot generates job descriptions and case studies based on uploaded documents.")

# 🔹 Cache Document Processing
@st.cache_resource
def load_documents():
    st.sidebar.write("Loading job descriptions & case studies...")
    documents = SimpleDirectoryReader(input_dir=".").load_data()
    Settings.embed_model = OpenAIEmbedding()
    index = VectorStoreIndex.from_documents(documents)
    st.sidebar.success("✅ Documents Loaded!")
    return index

index = load_documents()
query_engine = index.as_query_engine()

# 🔹 User Input Options
mode = st.radio("What do you want to generate?", ("Job Description", "Case Study"))
input_text = st.text_area("Enter the job role or case study topic:")

if st.button("Generate"):
    if not input_text:
        st.warning("Please enter a job role or case study topic!")
    else:
        with st.spinner("Generating content... ⏳"):
            # 🔹 Query the indexed documents for relevant content
            response = query_engine.query(input_text)

            # 🔹 Create a GPT-4o prompt based on the selected mode
            if mode == "Job Description":
                prompt = f"""
                You are an expert HR consultant generating job descriptions for Rohlik Group.

                **Standard Rohlik Group Job Description Format:**
                # Job Title: {input_text}

                ## About Rohlik Group  
                Rohlik Group is a leading online grocery company dedicated to delivering fresh, high-quality food with unbeatable convenience. We are innovating the grocery industry through cutting-edge technology, automated warehouses, and outstanding customer service.

                ## Role Overview  
                [Generate a professional role overview based on the context below.]

                ## Key Responsibilities  
                [List the key job responsibilities based on the context below.]

                ## Qualifications & Skills  
                [List essential qualifications and skills based on the context below.]

                ## Why Join Rohlik Group?  
                - Work with a fast-growing, tech-driven company  
                - Career development opportunities in a dynamic environment  
                - Competitive salary & benefits  

                **Context from company documents:**  
                {response}

                Please fill in the missing sections with structured, professional content.
                """

            elif mode == "Case Study":
                prompt = f"""
                You are a business analyst creating a case study for Rohlik Group. Follow this format:

                # Case Study: {input_text}

                ## About Rohlik Group  
                Rohlik Group is a leading online grocery company dedicated to delivering fresh, high-quality food with unbeatable convenience. We are innovating the grocery industry through cutting-edge technology, automated warehouses, and outstanding customer service.

                ## Business Challenge  
                [Generate a professional business challenge based on the context below.]

                ## Solution Implemented  
                [Describe the solution based on the context below.]

                ## Results & Impact  
                [Summarize the key outcomes based on the context below.]

                **Context from company documents:**  
                {response}

                Please complete the case study with well-structured, professional content.
                """


            # 🔹 Get GPT-4o to generate the content
            chat_response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": prompt}]
            )

            # 🔹 Display the output
            st.subheader("📢 Generated Content:")
            st.write(chat_response.choices[0].message.content)

            # Add a button to download the generated content as a PDF
            if st.button("Download as PDF"):
                pdf_buffer = generate_pdf(chat_response.choices[0].message.content)
                st.download_button(
                    label="📥 Click to Download PDF",
                    data=pdf_buffer,
                    file_name=f"{mode.replace(' ', '_')}.pdf",
                    mime="application/pdf"
    )

