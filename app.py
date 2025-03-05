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
            prompt = f"""
            You are an expert HR consultant and business strategist. 
            Generate a detailed {'job description' if mode == 'Job Description' else 'case study'} using the provided document context. 
            If additional details are needed, supplement with general industry knowledge.

            **Context from company documents:**
            {response}

            **Generate a {'job description' if mode == 'Job Description' else 'case study'} for:** {input_text}
            """

            # 🔹 Get GPT-4o to generate the content
            chat_response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": prompt}]
            )

            # 🔹 Display the output
            st.subheader("📢 Generated Content:")
            st.write(chat_response.choices[0].message.content)
