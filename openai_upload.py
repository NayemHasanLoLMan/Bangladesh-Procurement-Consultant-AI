import os
from pathlib import Path
import time
import hashlib
from pinecone import Pinecone, ServerlessSpec
import PyPDF2
import easyocr
from pdf2image import convert_from_path
from dotenv import load_dotenv
from openai import OpenAI

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

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536  # text-embedding-3-large dimension
BATCH_SIZE = 100
MAX_EMBEDDING_LENGTH = 8191  # OpenAI's token limit for text-embedding-3-large


class PDFUploader:
    def __init__(self, index_name="test-openai-singlefile-docs"):
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
            self._wait_until_ready(timeout=2)
        self.index = self.pc.Index(self.index_name)

    def _wait_until_ready(self, timeout=2, poll=2.0):
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
        """Get embedding using OpenAI with rate limiting"""
        max_retries = 3
        retry_delay = 5  # seconds
        
        for attempt in range(max_retries):
            try:
                # Generate embedding using OpenAI
                response = client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=text
                )
                
                # OpenAI has higher rate limits, but add small delay for safety
                time.sleep(0.1)
                
                return response.data[0].embedding
                
            except Exception as e:
                if "429" in str(e) or "rate_limit" in str(e).lower():
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (attempt + 1)
                        print(f" Rate limit hit, waiting {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        print(f" Embedding error after {max_retries} attempts: {e}")
                        return None
                else:
                    print(f" Embedding error: {e}")
                    return None
        
        return None
    
    def chunk_text(self, text, max_tokens=8000, overlap=200):
        """Split text into overlapping chunks based on token estimate"""
        # Rough estimate: 1 token ≈ 4 characters for English
        max_chars = max_tokens * 4
        overlap_chars = overlap * 4
        
        if len(text) <= max_chars:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + max_chars
            
            # Try to break at sentence or word boundary
            if end < len(text):
                # Look for sentence end
                last_period = text.rfind('.', start, end)
                last_question = text.rfind('?', start, end)
                last_exclaim = text.rfind('!', start, end)
                
                break_point = max(last_period, last_question, last_exclaim)
                
                # If no sentence boundary, try word boundary
                if break_point <= start:
                    last_space = text.rfind(' ', start, end)
                    if last_space > start:
                        break_point = last_space
                    else:
                        break_point = end
                else:
                    break_point += 1  # Include the punctuation
                
                end = break_point
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - overlap_chars
            if start < 0:
                start = end
        
        return chunks

    def generate_page_id(self, file_path, page_num, chunk_num=0):
        """Generate unique ID for each page chunk"""
        file_hash = hashlib.md5(file_path.encode("utf-8")).hexdigest()[:8]
        if chunk_num > 0:
            return f"{file_hash}_page_{page_num:04d}_chunk_{chunk_num:02d}"
        return f"{file_hash}_page_{page_num:04d}"

    def upload_pdf(self, pdf_path):
        """Upload PDF with chunking support for long pages"""
        print(f"Processing: {os.path.basename(pdf_path)}")
        
        pages_data = self.extract_text_from_pdf(pdf_path)
        if not pages_data:
            print(f" No text extracted\n")
            return
        
        print(f" Extracted {len(pages_data)} pages")
        
        vectors = []
        total_chunks = 0
        
        for page_info in pages_data:
            page_text = page_info["text"]
            page_num = page_info["page_num"]
            
            # Split page into chunks if needed
            chunks = self.chunk_text(page_text)
            
            for chunk_idx, chunk in enumerate(chunks):
                emb = self.get_embedding(chunk)
                if emb is None:
                    continue
                
                # Generate unique ID for each chunk
                if len(chunks) > 1:
                    vid = self.generate_page_id(pdf_path, page_num, chunk_idx + 1)
                else:
                    vid = self.generate_page_id(pdf_path, page_num)
                
                # Metadata with chunk info
                metadata = {
                    "source": os.path.basename(pdf_path),
                    "page": page_num,
                    "text": chunk[:5000]  # Store first 3000 chars for preview
                }
                
                if len(chunks) > 1:
                    metadata["chunk"] = chunk_idx + 1
                    metadata["total_chunks"] = len(chunks)
                
                vectors.append({
                    "id": vid,
                    "values": emb,
                    "metadata": metadata
                })
                
                total_chunks += 1
                
                # Batch upload
                if len(vectors) >= BATCH_SIZE:
                    self.index.upsert(vectors=vectors)
                    print(f"  → Uploaded {len(vectors)} vectors")
                    vectors = []
        
        # Upload remaining vectors
        if vectors:
            self.index.upsert(vectors=vectors)
            print(f" Uploaded {len(vectors)} vectors")
        
        print(f" Completed ({total_chunks} total chunks)\n")


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
    uploader = PDFUploader(index_name="test-openai-singlefile-docs")
    uploader.create_index()
    uploader.upload_directory(r"C:\Users\hasan\Downloads\PPR\documents-singlefile")