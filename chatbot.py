import openai
import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.settings import Settings  # New settings approach

# Ensure API key is set
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # Updated API client

# Configure LlamaIndex to use OpenAI Embeddings
Settings.embed_model = OpenAIEmbedding()  
Settings.llm = OpenAI(model="gpt-4o")  

# Load and process documents
documents = SimpleDirectoryReader(input_dir=".").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

# Start chatbot loop
print("\nChatbot is ready! Ask questions about your documents.\n")

while True:
    query = input("Ask me a question (or type 'exit' to quit): ")
    if query.lower() == "exit":
        print("Goodbye!")
        break

    # Query the indexed documents
    response = query_engine.query(query)

    # Use GPT-4o to refine the answer with the new OpenAI API format
    completion = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert in analyzing documents and answering questions."},
            {"role": "user", "content": f"Based on the following document context, answer the question:\n\n{response}\n\nQuestion: {query}"}
        ]
    )

    # Print the final answer
    print("\n" + completion.choices[0].message.content + "\n")
