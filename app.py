import streamlit as st
import openai
import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.settings import Settings

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
                You are a business consultant creating a structured case study for Rohlik Group. 
                This case study will be used to assess candidates' problem-solving abilities. 
                The case study **must** follow the structured format below.

                # 📌 Case Study: {input_text}

                ## **Company Overview**  
                Rohlik Group is a leading online grocery company dedicated to delivering fresh, high-quality food with unbeatable convenience. We are innovating the grocery industry through cutting-edge technology, automated warehouses, and outstanding customer service.

                ## **Business Challenge**  
                Describe a real-world business problem related to {input_text}. 
                The problem should be specific, measurable, and relevant to Rohlik Group’s operations. 
                Example structure:  
                - The **main issue** Rohlik Group faced  
                - The **context** in which the issue occurred  
                - The **implications** of not solving the problem  

                ## **Candidate Task**  
                The candidate must provide a structured analysis by answering these key questions:
                1️⃣ **How would you approach solving this problem?**  
                2️⃣ **What key data points or research would you gather?**  
                3️⃣ **What possible solutions would you consider?**  
                4️⃣ **How would you measure the success of your solution?**  
                5️⃣ **What are the risks and trade-offs of your approach?**  

                ## **Expected Deliverables from Candidate**  
                - A structured 1-2 page written response  
                - A PowerPoint slide with a high-level summary of the recommended solution  
                - Supporting data, charts, or research (if applicable)  

                ## **Solution Implemented by Rohlik Group**  
                Based on the retrieved document context, describe the **actual solution** Rohlik Group implemented to address this challenge.  

                ## **Results & Impact**  
                Provide quantifiable metrics on how the solution improved operations, revenue, efficiency, or customer satisfaction.

                ---

                Please generate a case study following this structure. Ensure the output is actionable and ready to be given to candidates.
                """



            # 🔹 Get GPT-4o to generate the content
            chat_response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": prompt}]
            )

            # 🔹 Display the output
            st.subheader("📢 Generated Content:")
            st.write(chat_response.choices[0].message.content)


    

