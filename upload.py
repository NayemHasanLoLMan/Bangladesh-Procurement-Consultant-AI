import os
from pathlib import Path
import time
import hashlib
import re
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import PyPDF2
import easyocr
from pdf2image import convert_from_path
from PIL import Image
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

EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSION = 3072
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
BATCH_SIZE = 100
MAX_EMBEDDING_LENGTH = 8191  # OpenAI's actual token limit for embeddings


class PDFUploader:
    def __init__(self, index_name="bangladesh-procurement-docs"):
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index_name = index_name
        self.index = None
        
        # Initialize EasyOCR with English and Bengali
        print("Initializing OCR (this may take a moment)...")
        self.ocr_reader = easyocr.Reader(['en', 'bn'], gpu=False)
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

    def detect_language(self, text):
        """Detect if text contains Bangla characters"""
        bangla_pattern = re.compile(r'[\u0980-\u09FF]')
        has_bangla = bool(bangla_pattern.search(text))
        has_english = bool(re.search(r'[a-zA-Z]', text))
        
        if has_bangla and has_english:
            return "mixed"
        elif has_bangla:
            return "bangla"
        else:
            return "english"

    def detect_table(self, text):
        """Simple table detection based on patterns"""
        lines = text.split('\n')
        pipe_count = sum(1 for line in lines if '|' in line or '\t' in line)
        aligned_spaces = sum(1 for line in lines if re.search(r'\s{3,}', line))
        
        return (pipe_count > 2) or (aligned_spaces > len(lines) * 0.3)

    def extract_text_with_ocr(self, pdf_path):
        """Extract text using OCR for image-heavy PDFs"""
        try:
            images = convert_from_path(pdf_path, dpi=200)
            ocr_data = []
            
            for page_num, image in enumerate(images):
                # Perform OCR
                results = self.ocr_reader.readtext(image)
                
                # Extract text from OCR results
                page_text = "\n".join([text for (_, text, _) in results])
                
                if page_text.strip():
                    ocr_data.append({
                        "page_num": page_num + 1,
                        "text": page_text,
                        "extracted_by": "ocr"
                    })
                    
            return ocr_data
        except Exception as e:
            print(f"  ⚠ OCR extraction failed: {e}")
            return []

    def extract_text_from_pdf(self, pdf_path):
        """Extract text with page and structure metadata, fallback to OCR if needed"""
        pages_data = []
        
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                
                # If page has very little text, it might be an image/scan
                if len(page_text.strip()) < 50:
                    continue
                
                if page_text.strip():
                    language = self.detect_language(page_text)
                    has_table = self.detect_table(page_text)
                    
                    pages_data.append({
                        "page_num": page_num + 1,
                        "text": page_text,
                        "language": language,
                        "has_table": has_table,
                        "extracted_by": "pypdf"
                    })
        
        # If very few pages extracted, try OCR
        if len(pages_data) < 3:
            print(f"  → Low text extraction, attempting OCR...")
            ocr_pages = self.extract_text_with_ocr(pdf_path)
            
            # Merge OCR results with existing pages
            for ocr_page in ocr_pages:
                page_num = ocr_page["page_num"]
                
                # Check if this page already exists
                existing = next((p for p in pages_data if p["page_num"] == page_num), None)
                
                if not existing or len(existing["text"]) < 50:
                    # Add language and table detection
                    ocr_page["language"] = self.detect_language(ocr_page["text"])
                    ocr_page["has_table"] = self.detect_table(ocr_page["text"])
                    pages_data.append(ocr_page)
        
        # Sort by page number
        pages_data.sort(key=lambda x: x["page_num"])
        return pages_data

    def normalize_text(self, text):
        """Clean and normalize text while preserving structure"""
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def smart_chunk_text(self, pages_data, base_metadata):
        """Create semantically meaningful chunks without losing text"""
        chunks = []
        
        for page_info in pages_data:
            text = self.normalize_text(page_info["text"])
            page_num = page_info["page_num"]
            language = page_info["language"]
            has_table = page_info["has_table"]
            extracted_by = page_info.get("extracted_by", "pypdf")
            
            # If page has table and is small enough, keep intact
            if has_table and len(text) <= CHUNK_SIZE * 1.5:
                chunk_meta = {
                    **base_metadata,
                    "page_number": page_num,
                    "language": language,
                    "content_type": "table",
                    "chunk_index": len(chunks),
                    "extracted_by": extracted_by
                }
                chunks.append({
                    "text": text,
                    "metadata": chunk_meta
                })
                continue
            
            # Regular chunking - NO TEXT LOST
            start = 0
            n = len(text)
            
            while start < n:
                end = min(start + CHUNK_SIZE, n)
                chunk = text[start:end]
                
                # Find natural break points
                if end < n:
                    last_para = chunk.rfind('\n\n')
                    last_period = chunk.rfind('. ')
                    last_bangla_stop = chunk.rfind('।')
                    last_space = chunk.rfind(' ')
                    
                    breaks = [
                        (last_para, 2) if last_para > CHUNK_SIZE * 0.5 else (0, 0),
                        (last_period, 2) if last_period > CHUNK_SIZE * 0.7 else (0, 0),
                        (last_bangla_stop, 1) if last_bangla_stop > CHUNK_SIZE * 0.7 else (0, 0),
                        (last_space, 0) if last_space > CHUNK_SIZE * 0.8 else (0, 0)
                    ]
                    
                    for pos, offset in breaks:
                        if pos > 0:
                            end = start + pos + offset
                            chunk = text[start:end]
                            break
                
                if chunk.strip():
                    chunk_meta = {
                        **base_metadata,
                        "page_number": page_num,
                        "language": language,
                        "content_type": "table" if has_table else "text",
                        "chunk_index": len(chunks),
                        "char_count": len(chunk),
                        "extracted_by": extracted_by
                    }
                    chunks.append({
                        "text": chunk.strip(),
                        "metadata": chunk_meta
                    })
                
                # Smart overlap to ensure no text is lost
                overlap = CHUNK_OVERLAP // 2 if has_table else CHUNK_OVERLAP
                start = end - overlap if end < n else n
        
        return chunks

    def get_embedding(self, text):
        """Get embedding with proper chunking for long texts"""
        try:
            # Rough estimate: 1 token ≈ 4 characters
            estimated_tokens = len(text) // 4
            
            # If text is too long, chunk it and average embeddings
            if estimated_tokens > MAX_EMBEDDING_LENGTH:
                # Split into smaller parts
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
            
            # Normal case: text fits in one embedding
            resp = self.openai_client.embeddings.create(
                input=text,
                model=EMBEDDING_MODEL,
            )
            return resp.data[0].embedding
            
        except Exception as e:
            print(f"  ⚠ Embedding error: {e}")
            return None

    def generate_chunk_id(self, file_path, chunk_index):
        file_hash = hashlib.md5(file_path.encode("utf-8")).hexdigest()[:8]
        return f"{file_hash}_chunk_{chunk_index:04d}"

    def upload_pdf(self, pdf_path, custom_metadata=None):
        """Upload PDF with enhanced organization"""
        print(f"Processing: {os.path.basename(pdf_path)}")
        
        pages_data = self.extract_text_from_pdf(pdf_path)
        if not pages_data:
            print(f"  ⚠ No text extracted\n")
            return
        
        # Detect overall document language
        all_languages = [p["language"] for p in pages_data]
        doc_language = max(set(all_languages), key=all_languages.count)
        
        # Count extraction methods
        ocr_pages = sum(1 for p in pages_data if p.get("extracted_by") == "ocr")
        
        base_metadata = {
            "source": os.path.basename(pdf_path),
            "file_path": pdf_path,
            "document_type": "procurement_document",
            "country": "Bangladesh",
            "total_pages": len(pages_data),
            "primary_language": doc_language,
            "ocr_pages": ocr_pages
        }
        
        if custom_metadata:
            base_metadata.update(custom_metadata)
        
        chunks = self.smart_chunk_text(pages_data, base_metadata)
        print(f"  ✓ Created {len(chunks)} chunks ({doc_language})")
        if ocr_pages > 0:
            print(f"  ✓ OCR used on {ocr_pages} pages")
        
        vectors = []
        for i, chunk in enumerate(chunks):
            emb = self.get_embedding(chunk["text"])
            if emb is None:
                continue
            
            vid = self.generate_chunk_id(pdf_path, i)
            
            # Store reasonable text preview in metadata
            text_preview = chunk["text"][:500]
            
            vectors.append({
                "id": vid,
                "values": emb,
                "metadata": {
                    **chunk["metadata"],
                    "text": text_preview
                }
            })
            
            if len(vectors) >= BATCH_SIZE:
                self.index.upsert(vectors=vectors)
                print(f"  → Uploaded {len(vectors)} vectors")
                vectors = []
        
        if vectors:
            self.index.upsert(vectors=vectors)
            print(f"  → Uploaded {len(vectors)} vectors")
        
        print(f"  Completed\n")

    def upload_directory(self, directory_path):
        """Upload all PDFs in directory"""
        pdf_files = list(Path(directory_path).glob("*.pdf"))
        print(f"Found {len(pdf_files)} PDF files\n")
        
        for pdf_file in pdf_files:
            try:
                self.upload_pdf(str(pdf_file))
            except Exception as e:
                print(f"  Error: {e}\n")


if __name__ == "__main__":
    uploader = PDFUploader(index_name="ppr-documents")
    uploader.create_index()
    uploader.upload_directory(r"C:/Users/USER/Downloads/Documents/split pdf")