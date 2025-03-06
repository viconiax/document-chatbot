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
                You are designing a structured case study for candidates applying to Rohlik Group. 
                The case study should test problem-solving, data analysis, and strategic thinking. 
                **Do NOT provide the answer**—the goal is to present the challenge for the candidate.

                # 📌 Case Study: {input_text}

                ## **Company Overview**  
                Rohlik Group is a leading online grocery company delivering fresh, high-quality food with unbeatable convenience. We are transforming the grocery industry through cutting-edge technology, automated warehouses, and outstanding customer service.

                ## **Business Challenge**  
                Provide a **brief, structured problem statement** relevant to {input_text}.  
                - 🔹 **Main issue:** Describe the core challenge in 2-3 sentences.  
                - 🔹 **Context:** What situation led to this issue? (Keep it short, max 2 sentences)  
                - 🔹 **Implications:** What risks or impacts does this challenge create? (Bullet points, 2-3 max)  

                ## **Candidate Task**  
                The candidate must develop a structured response addressing the problem by answering:  
                1️⃣ **How would you approach solving this issue?**  
                2️⃣ **What key data points or market research would you use?**  
                3️⃣ **What possible solutions would you consider?**  
                4️⃣ **How would you evaluate the effectiveness of your solution?**  
                5️⃣ **What risks and trade-offs should Rohlik Group be aware of?**  

                ## **Expected Deliverables**  
                - A structured **1-2 page written response** outlining the proposed solution  
                - A **PowerPoint slide** summarizing key recommendations  
                - Supporting **data, research, or cost-benefit analysis** (if applicable)  

                """



            # 🔹 Get GPT-4o to generate the content
            chat_response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": prompt}]
            )

            # 🔹 Display the output
            st.subheader("📢 Generated Content:")
            st.write(chat_response.choices[0].message.content)


    

