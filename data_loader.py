import os
import re
from langchain_community.document_loaders import PyPDFLoader

def clean_text(text):
    """
    Cleans the input text by:
    - Removing non-printable and control characters
    - Normalizing whitespace (tabs/spaces to single space)
    - Normalizing newlines (multiple newlines to single newline)
    - Stripping leading/trailing whitespace
    """
    if not text:
        return ""
    
    # Remove non-printable characters but keep newlines and tabs
    text = "".join(char for char in text if char.isprintable() or char in "\n\t")
    
    # Replace multiple spaces/tabs with a single space
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Remove empty lines / normalize multiple newlines to single newlines
    text = re.sub(r'\n+', '\n', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text

def load_and_clean_pdfs(directory_path):
    """
    Loads all PDF files from the given directory and cleans their content.
    """
    all_documents = []
    pdf_files = [f for f in os.listdir(directory_path) if f.lower().endswith('.pdf')]
    
    print(f"Found {len(pdf_files)} PDF files: {', '.join(pdf_files)}")
    
    for pdf_file in pdf_files:
        file_path = os.path.join(directory_path, pdf_file)
        try:
            print(f"Loading {pdf_file}...")
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            
            for doc in docs:
                # Clean the page content
                doc.page_content = clean_text(doc.page_content)
                # Ensure we don't include empty documents after cleaning
                if doc.page_content:
                    all_documents.append(doc)
                    
        except Exception as e:
            print(f"Error loading {pdf_file}: {e}")
            
    return all_documents

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load and clean documents
    documents = load_and_clean_pdfs(current_dir)
    
    # Output statistics
    print("-" * 20)
    print(f"Total documents (pages) loaded and cleaned: {len(documents)}")
    
    # Print a sample cleaned document (first page of the first document)
    if documents:
        print("\n--- Sample Cleaned Document (First 500 characters) ---")
        sample_doc = documents[0]
        print(f"Source: {sample_doc.metadata.get('source', 'Unknown')}")
        print(f"Content Sample:\n{sample_doc.page_content[:500]}...")
        print("-" * 20)
    else:
        print("No documents found or loaded.")
