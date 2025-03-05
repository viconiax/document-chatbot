import streamlit as st
import openai
import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.settings import Settings

# 🔹 OpenAI API Key Setup
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # Make sure to set your API Key!

# 🔹 Load and index your documents
st.sidebar.title("📄 Document Chatbot")
st.sidebar.write("Loading documents...")

documents = SimpleDirectoryReader(input_dir=".").load_data()
Settings.embed_model = OpenAIEmbedding()  
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

st.sidebar.success("✅ Documents Loaded!")

# 🔹 Streamlit UI
st.title("💬 Chat with Your Documents")
st.write("Ask any question about the uploaded documents!")

query = st.text_input("Enter your question:")

if st.button("Ask"):
    if not query:
        st.warning("Please enter a question!")
    else:
        with st.spinner("Thinking... 🤔"):
            response = query_engine.query(query)

            # 🔹 Use GPT-4o to refine the answer
            chat_response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an AI assistant that answers questions based on uploaded documents."},
                    {"role": "user", "content": f"Here is the document context:\n{response}\n\nQuestion: {query}"}
                ]
            )
            
            st.subheader("📢 Answer:")
            st.write(chat_response.choices[0].message.content)
