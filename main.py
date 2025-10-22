import os
import re
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"  # Better multilingual support than gpt-4-turbo


class ProcurementConsultant:
    def __init__(self, index_name="test-docs"):
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index = self.pc.Index(index_name)

        # Language-aware system prompts
        self.bangla_prompt = """আপনি বাংলাদেশ সরকারী ক্রয় কর্তৃপক্ষ (BPPA) বিধি, নিয়মকানুন এবং পদ্ধতির একজন বিশেষজ্ঞ পরামর্শদাতা। আপনার দায়িত্ব হলো সংগ্রহ সংক্রান্ত বিষয়ে স্পষ্ট, ব্যবহারিক এবং নির্ভুল পরামর্শ প্রদান করা।

        আপনার ভূমিকা:
        - BPPA নথি থেকে প্রাপ্ত তথ্যের সাথে আপনার বিশেষজ্ঞ জ্ঞান সমন্বয় করে বিস্তারিত এবং কার্যকর উত্তর প্রদান করুন
        - ব্যবহারকারীর প্রশ্নের সরাসরি এবং পূর্ণাঙ্গ সমাধান দিন
        - নথিতে তথ্য না থাকলে, BPPA নিয়মকানুন সম্পর্কে আপনার জ্ঞান ব্যবহার করে নির্ভুল পরামর্শ দিন
        - সবসময় বাংলায় উত্তর দিন যখন প্রশ্ন বাংলায় করা হয়
        - এটা উল্লেখ করবে না যে তুমি একটি কৃত্রিম বুদ্ধিমত্তা বা ভাষা মডেল এবং তোমার কোনো সীমাবদ্ধতা আছে।

        নির্দেশনা:
        1. **ব্যবহারিক এবং কার্যকর**: উত্তর এমন হতে হবে যা ব্যবহারকারী সহজেই প্রয়োগ করতে পারেন
        2. **স্পষ্ট এবং সম্পূর্ণ**: যতটা সম্ভব বিস্তারিত তথ্য প্রদান করুন
        3. **পেশাদার এবং সহজবোধ্য**: জটিল আইনি ভাষা এড়িয়ে সাধারণ বাংলায় ব্যাখ্যা করুন
        4. **নির্ভুল এবং সত্য**: শুধুমাত্র BPPA নিয়ম এবং নথির তথ্যের উপর ভিত্তি করে উত্তর দিন

        প্রাসঙ্গিক নথি থেকে তথ্য:
        {context}

        দয়া করে বাংলায় বিস্তারিত এবং নির্ভুল উত্তর প্রদান করুন।"""

        self.english_prompt = """You are an expert consultant specializing in Bangladesh Public Procurement Authority (BPPA) regulations, rules, and procedures. Your role is to provide clear, practical, and accurate guidance on procurement-related matters.

        YOUR ROLE:
        - Provide detailed answers by combining information from BPPA documents with your expert knowledge
        - Address user questions directly with comprehensive and actionable solutions
        - If documents don't contain explicit answers, use your knowledge of BPPA regulations to provide well-informed advice
        - Always respond in English when the query is in English
        - Do NOT mention that you are an AI or language model and your limitaion

        INSTRUCTIONS:
        1. **Prioritize Practical Guidance**: Deliver actionable answers that solve the user's problem
        2. **Ensure Clarity and Completeness**: Provide as much relevant detail as possible
        3. **Professional and Clear Tone**: Use professional but accessible language
        4. **Accurate Information**: Base your answers on BPPA regulations and the provided documents

        Relevant document information:
        {context}

        Please provide a detailed and accurate answer in English."""

    def detect_language(self, text):
        """Detect if query is in Bangla or English"""
        bangla_chars = re.findall(r'[\u0980-\u09FF]', text)
        english_chars = re.findall(r'[a-zA-Z]', text)
        
        bangla_count = len(bangla_chars)
        english_count = len(english_chars)
        
        # If more than 30% Bangla characters, consider it Bangla
        if bangla_count > english_count * 0.5:
            return "bangla"
        return "english"

    def get_embedding(self, text):
        """Generate embedding for text"""
        response = self.openai_client.embeddings.create(
            input=text,
            model=EMBEDDING_MODEL
        )
        return response.data[0].embedding
    
    def search_documents(self, query, top_k=5):
        """Search vector database for relevant documents"""
        query_embedding = list(self.get_embedding(query))
        
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        return [
            {
                "text": match.metadata.get('text', ''),
                "score": match.score,
                "source": match.metadata.get('source', 'Unknown'),
                "page": match.metadata.get('page', 'N/A')
            }
            for match in results.matches
        ]
    
    def build_context(self, results, is_bangla=False):
        """Build context from search results with language-aware token management"""
        if not results:
            return "No relevant documents found."
        
        # Allocate more tokens for Bangla (Bangla uses ~1.5-2x more tokens)
        max_tokens = 2500 if is_bangla else 4000
        max_chars = max_tokens * 3 if is_bangla else max_tokens * 4
        
        context_parts = []
        total_chars = 0
        
        for i, match in enumerate(results, 1):
            text = match['text']
            source = match['source']
            page = match['page']
            
            if not text:
                continue
            
            header = f"\n[Document {i} - {source}, Page {page}]\n"
            
            # Calculate remaining space
            remaining_chars = max_chars - total_chars - len(header)
            
            if remaining_chars <= 100:
                break
            
            # Truncate text if needed
            text_chunk = text[:remaining_chars]
            
            context_parts.append(header + text_chunk)
            total_chars += len(header) + len(text_chunk)
        
        return "\n".join(context_parts)
    
    def consult(self, user_input):
        """Main consultation method with language detection"""
        
        # Detect query language
        query_language = self.detect_language(user_input)
        is_bangla = (query_language == "bangla")
        
        # Search for relevant documents
        results = self.search_documents(user_input, top_k=5)
        
        # Build context with language awareness
        context = self.build_context(results, is_bangla=is_bangla)
        
        # Select appropriate prompt
        system_prompt = self.bangla_prompt if is_bangla else self.english_prompt
        system_prompt = system_prompt.format(context=context)
        
        # Adjust max_tokens based on language
        # Bangla needs more tokens for same content
        max_tokens = 2000 if is_bangla else 1500
        
        # Build messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
        
        # Generate response
        try:
            response = self.openai_client.chat.completions.create(
                model=CHAT_MODEL,
                messages=messages,
                temperature=0.3,  # Lower for more factual responses
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            if is_bangla:
                return f"দুঃখিত, একটি ত্রুটি হয়েছে: {str(e)}"
            return error_msg
    
    def consult_with_details(self, user_input):
        """Consultation with source information"""
        
        # Detect language
        query_language = self.detect_language(user_input)
        is_bangla = (query_language == "bangla")
        
        # Search documents
        results = self.search_documents(user_input, top_k=5)
        
        # Get answer
        answer = self.consult(user_input)
        
        # Format sources
        sources = []
        for match in results:
            sources.append({
                'source': match['source'],
                'page': match['page'],
                'relevance_score': round(match['score'], 4),
                'text_preview': match['text'][:300] + "..."
            })
        
        return {
            'answer': answer,
            'language': query_language,
            'sources': sources,
            'total_sources_used': len(sources)
        }


def get_consultation(text_input):
    """Wrapper function for easy use"""
    consultant = ProcurementConsultant(index_name="test-docs")
    result = consultant.consult_with_details(text_input)
    
    return {
        "answer": result.get('answer'),
        "language": result.get('language'),
        # Uncomment if you want to see sources
        # "sources": result.get('sources', []),
        # "total_sources_used": result.get('total_sources_used', 0),
    }


if __name__ == "__main__":

    # answer = get_consultation(
    #     text_input="ভৌত সেবা সংক্রান্ত চুক্তি ব্যবস্থাপনা সম্পর্কে বিস্তারিত বলুন"
    # )

    # print(f"Answer:\n{answer['answer']}\n")
    

    answer = get_consultation(
        text_input="which one should from first TEC or TOC?"
    )

    print(f"Answer:\n{answer['answer']}")