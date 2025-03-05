from docx import Document
import os

# Path to folder with Word docs
folder_path = "word_docs"
output_file = "documents.txt"

with open(output_file, "w", encoding="utf-8") as outfile:
    for filename in os.listdir(folder_path):
        if filename.endswith(".docx"):  # Only process .docx files
            doc = Document(os.path.join(folder_path, filename))
            text = "\n".join([para.text for para in doc.paragraphs])
            outfile.write(f"=== {filename} ===\n{text}\n\n")

print("Extraction complete! Text saved in documents.txt")
