import os
from pathlib import Path
import time
import hashlib
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import PyPDF2
import easyocr
from pdf2image import convert_from_path
from dotenv import load_dotenv

load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_REGION = os.getenv("PINECONE_REGION") or os.getenv("PINECONE_ENVIRONMENT") or "us-east-1"

# Validate environment variables
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment")

print(f"Using Pinecone region: {PINECONE_REGION}")

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536
BATCH_SIZE = 100
MAX_EMBEDDING_LENGTH = 8191  # OpenAI's token limit


class PDFUploader:
    def __init__(self, index_name="Test-docs"):
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index_name = index_name
        self.index = None
        
        # Initialize EasyOCR with English and Bengali
        print("Initializing OCR (this may take a moment)...")
        self.ocr_reader = easyocr.Reader(['en', 'bn'], gpu=True)
        print("OCR ready!\n")

    def create_index(self):
        existing = self.pc.list_indexes().names()
        if self.index_name not in existing:
            self.pc.create_index(
                name=self.index_name,
                dimension=EMBEDDING_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION),
            )
            self._wait_until_ready(timeout=180)
        self.index = self.pc.Index(self.index_name)

    def _wait_until_ready(self, timeout=180, poll=2.0):
        start = time.time()
        while True:
            desc = self.pc.describe_index(self.index_name)
            status = getattr(desc, "status", None)
            ready = getattr(status, "ready", None)
            state = getattr(status, "state", None)
            if ready is True or state == "Ready":
                return
            if time.time() - start > timeout:
                raise TimeoutError(f"Index {self.index_name} not ready after {timeout}s")
            time.sleep(poll)

    def extract_text_with_ocr(self, pdf_path):
        """Extract text using OCR for image-heavy PDFs"""
        try:
            images = convert_from_path(pdf_path, dpi=200)
            ocr_data = []
            
            for page_num, image in enumerate(images):
                results = self.ocr_reader.readtext(image)
                page_text = "\n".join([text for (_, text, _) in results])
                
                if page_text.strip():
                    ocr_data.append({
                        "page_num": page_num + 1,
                        "text": page_text
                    })
                    
            return ocr_data
        except Exception as e:
            print(f" OCR extraction failed: {e}")
            return []

    def extract_text_from_pdf(self, pdf_path):
        """Extract full page text without chunking"""
        pages_data = []
        
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                
                if page_text.strip() and len(page_text.strip()) >= 50:
                    pages_data.append({
                        "page_num": page_num + 1,
                        "text": page_text
                    })
        
        # If very few pages extracted, try OCR
        if len(pages_data) < 3:
            print(f" Low text extraction, attempting OCR...")
            ocr_pages = self.extract_text_with_ocr(pdf_path)
            
            for ocr_page in ocr_pages:
                page_num = ocr_page["page_num"]
                existing = next((p for p in pages_data if p["page_num"] == page_num), None)
                
                if not existing or len(existing["text"]) < 50:
                    pages_data.append(ocr_page)
        
        pages_data.sort(key=lambda x: x["page_num"])
        return pages_data

    def get_embedding(self, text):
        """Get embedding with proper chunking for long texts"""
        try:
            estimated_tokens = len(text) // 4
            
            # If text is too long, chunk it and average embeddings
            if estimated_tokens > MAX_EMBEDDING_LENGTH:
                max_chars = MAX_EMBEDDING_LENGTH * 4
                parts = [text[i:i+max_chars] for i in range(0, len(text), max_chars)]
                
                embeddings = []
                for part in parts:
                    resp = self.openai_client.embeddings.create(
                        input=part,
                        model=EMBEDDING_MODEL,
                    )
                    embeddings.append(resp.data[0].embedding)
                
                # Average the embeddings
                avg_embedding = [sum(x) / len(embeddings) for x in zip(*embeddings)]
                return avg_embedding
            
            # Normal case
            resp = self.openai_client.embeddings.create(
                input=text,
                model=EMBEDDING_MODEL,
            )
            return resp.data[0].embedding
            
        except Exception as e:
            print(f" Embedding error: {e}")
            return None

    def generate_page_id(self, file_path, page_num):
        """Generate unique ID for each page"""
        file_hash = hashlib.md5(file_path.encode("utf-8")).hexdigest()[:8]
        return f"{file_hash}_page_{page_num:04d}"

    def upload_pdf(self, pdf_path):
        """Upload PDF with one vector per page"""
        print(f"Processing: {os.path.basename(pdf_path)}")
        
        pages_data = self.extract_text_from_pdf(pdf_path)
        if not pages_data:
            print(f" No text extracted\n")
            return
        
        print(f" Extracted {len(pages_data)} pages")
        
        vectors = []
        for page_info in pages_data:
            page_text = page_info["text"]
            page_num = page_info["page_num"]
            
            emb = self.get_embedding(page_text)
            if emb is None:
                continue
            
            vid = self.generate_page_id(pdf_path, page_num)
            
            # Minimal metadata
            vectors.append({
                "id": vid,
                "values": emb,
                "metadata": {
                    "source": os.path.basename(pdf_path),
                    "page": page_num,
                    "text": page_text[:3000]  # Store first 1000 chars for preview
                }
            })
            
            if len(vectors) >= BATCH_SIZE:
                self.index.upsert(vectors=vectors)
                print(f"  → Uploaded {len(vectors)} vectors")
                vectors = []
        
        if vectors:
            self.index.upsert(vectors=vectors)
            print(f" Uploaded {len(vectors)} vectors")
        
        print(f" Completed\n")

    def upload_directory(self, directory_path):
        """Upload all PDFs in directory"""
        pdf_files = list(Path(directory_path).glob("*.pdf"))
        print(f"Found {len(pdf_files)} PDF files\n")
        
        for pdf_file in pdf_files:
            try:
                self.upload_pdf(str(pdf_file))
            except Exception as e:
                print(f" Error: {e}\n")


if __name__ == "__main__":
    uploader = PDFUploader(index_name="test-docs")
    uploader.create_index()
    uploader.upload_directory(r"C:\Users\hasan\Downloads\PPR\documents")