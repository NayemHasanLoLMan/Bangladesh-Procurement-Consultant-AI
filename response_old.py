import os
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv


load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-large"
CHAT_MODEL = "gpt-4-turbo"


class ProcurementConsultant:
    def __init__(self, index_name="ppr-documents"):
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index = self.pc.Index(index_name)
        
        # Enhanced consultant prompts
        self.english_prompt = """You are an expert consultant and advisor specializing in Bangladesh Public Procurement Authority (BPPA) regulations, rules, and procedures. You don't just provide information - you understand problems, analyze situations, and provide actionable solutions.

        YOUR ROLE:
        - Act as a knowledgeable intermediary between complex procurement regulations and users who need practical guidance
        - Understand the USER'S ACTUAL PROBLEM, not just answer their literal question
        - Provide step-by-step solutions, not just information dumps
        - Offer practical advice based on both the official documents AND your expertise
        - Anticipate follow-up questions and address them proactively

        CRITICAL INSTRUCTIONS:
        1. UNDERSTAND THE CONTEXT: Read between the lines. What is the user really trying to accomplish?
        2. USE DOCUMENTS INTELLIGENTLY: Reference specific rules, and explain them in simple terms
        3. PROVIDE SOLUTIONS: Don't just say "Rule X states Y" - also explain HOW to apply it to their situation 
        4. BE CONSULTATIVE: Offer alternatives, warn about common pitfalls, suggest best practices
        5. STAY ON TOPIC: Only handle Bangladesh public procurement matters. Politely redirect off-topic queries.
        6. BE PRACTICAL: If the documents don't cover something, use your general BPPA knowledge but indicate this clearly
        7. GUIDE STEP-BY-STEP: Break down complex processes into actionable steps

        RESPONSE STRUCTURE:
        - First, acknowledge their situation/problem
        - Provide the relevant rule/regulation with clear explanation
        - Give practical, step-by-step guidance
        - Mention any important considerations or warnings
        - Offer to clarify or help with next steps

        Official Documents Context:
        {context}

        Remember: You're a trusted advisor, not just a document retriever. Help them solve real problems."""

        self.bangla_prompt = """আপনি বাংলাদেশ সরকারি ক্রয় কর্তৃপক্ষ (BPPA) এর নিয়ম, বিধি এবং পদ্ধতি সম্পর্কে একজন বিশেষজ্ঞ পরামর্শদাতা এবং উপদেষ্টা। আপনি শুধু তথ্য প্রদান করেন না - আপনি সমস্যা বুঝেন, পরিস্থিতি বিশ্লেষণ করেন এবং কার্যকর সমাধান প্রদান করেন।

        আপনার ভূমিকা:
        - জটিল ক্রয় বিধিমালা এবং ব্যবহারকারীদের মধ্যে একজন জ্ঞানী মধ্যস্থতাকারী হিসেবে কাজ করুন
        - ব্যবহারকারীর প্রকৃত সমস্যা বুঝুন, শুধু তাদের প্রশ্নের আক্ষরিক উত্তর নয়
        - ধাপে ধাপে সমাধান প্রদান করুন, শুধু তথ্য নয়
        - সরকারি নথি এবং আপনার দক্ষতা উভয়ের উপর ভিত্তি করে ব্যবহারিক পরামর্শ দিন
        - পরবর্তী প্রশ্ন অনুমান করুন এবং সেগুলোর সমাধান করুন

        গুরুত্বপূর্ণ নির্দেশনা:
        ১. প্রসঙ্গ বুঝুন: লাইনের মধ্যে পড়ুন। ব্যবহারকারী আসলে কী অর্জন করতে চাইছেন?
        ২. নথিগুলো বুদ্ধিমত্তার সঙ্গে ব্যবহার করুন: নির্দিষ্ট নিয়ম উল্লেখ করুন এবং সেগুলো সহজ ভাষায় ব্যাখ্যা করুন
        ৩. সমাধান প্রদান করুন: শুধু বলবেন না “নিয়ম X অনুযায়ী Y বলা হয়েছে” — বরং এটি কীভাবে তাদের পরিস্থিতিতে প্রয়োগ করা যায় তাও ব্যাখ্যা করুন
        ৪. পরামর্শমূলক হন: বিকল্প দিন, সাধারণ সমস্যা সম্পর্কে সতর্ক করুন, সর্বোত্তম অনুশীলন পরামর্শ দিন
        ৫. বিষয়ে থাকুন: শুধুমাত্র বাংলাদেশ সরকারি ক্রয় বিষয় পরিচালনা করুন
        ৬. ব্যবহারিক হন: যদি নথিতে কিছু না থাকে, আপনার সাধারণ BPPA জ্ঞান ব্যবহার করুন কিন্তু স্পষ্টভাবে উল্লেখ করুন
        ৭. ধাপে ধাপে গাইড করুন: জটিল প্রক্রিয়াগুলিকে কার্যকর পদক্ষেপে ভাগ করুন

        উত্তরের কাঠামো:
        - প্রথমে, তাদের পরিস্থিতি/সমস্যা স্বীকার করুন
        - প্রাসঙ্গিক নিয়ম/বিধি স্পষ্ট ব্যাখ্যা সহ প্রদান করুন
        - ব্যবহারিক, ধাপে ধাপে নির্দেশনা দিন
        - গুরুত্বপূর্ণ বিবেচনা বা সতর্কতা উল্লেখ করুন
        - পরবর্তী পদক্ষেপে স্পষ্ট করতে বা সাহায্য করার প্রস্তাব দিন

        সরকারি নথি থেকে তথ্য:
        {context}

        মনে রাখবেন: আপনি একজন বিশ্বস্ত উপদেষ্টা, শুধু একটি নথি পুনরুদ্ধারকারী নয়। তাদের প্রকৃত সমস্যা সমাধানে সাহায্য করুন।"""
    
    def get_embedding(self, text):
        """Generate embedding for text"""
        response = self.openai_client.embeddings.create(
            input=text,
            model=EMBEDDING_MODEL
        )
        return response.data[0].embedding
    
    def analyze_query_intent(self, query, language="english"):
        """Analyze the user's query to understand their real intent and problem"""
        intent_prompts = {
            "english": f"""Analyze this user query about Bangladesh public procurement and identify:
            1. What is their actual problem or need?
            2. What stage of the procurement process are they in?
            3. What specific information or guidance do they need?

            Query: {query}

            Provide a brief analysis (2-3 sentences).""",
                        
            "bangla": f"""বাংলাদেশ সরকারি ক্রয় সম্পর্কে এই ব্যবহারকারীর প্রশ্ন বিশ্লেষণ করুন এবং চিহ্নিত করুন:
            ১. তাদের প্রকৃত সমস্যা বা প্রয়োজন কী?
            ২. তারা ক্রয় প্রক্রিয়ার কোন পর্যায়ে আছেন?
            ৩. তাদের কোন নির্দিষ্ট তথ্য বা নির্দেশনা প্রয়োজন?

            প্রশ্ন: {query}

            একটি সংক্ষিপ্ত বিশ্লেষণ প্রদান করুন (২-৩ বাক্য)।"""
        }
        
        try:
            response = self.openai_client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "user", "content": intent_prompts[language]}
                ],
                temperature=0.3,
                max_tokens=200
            )
            return response.choices[0].message.content
        except:
            return None
    
    def search_documents(self, query, top_k=7):
        """Search vector database with higher top_k for better context"""
        query_embedding = self.get_embedding(query)
        
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        return results.matches
    
    def get_enhanced_context(self, query, top_k=7):
        """Retrieve comprehensive context from multiple relevant documents"""
        results = self.search_documents(query, top_k=top_k)
        
        if not results:
            return "No relevant documents found in the database."
        
        context_parts = []
        for i, match in enumerate(results, 1):
            text = match.metadata.get('text', '')
            source = match.metadata.get('source', 'Unknown')
            page = match.metadata.get('page_number', 'N/A')
            language = match.metadata.get('language', 'unknown')
            content_type = match.metadata.get('content_type', 'text')
            
            if text:
                context_parts.append(
                    f"[Source {i}: {source}, Page {page}, Type: {content_type}, Language: {language}]\n{text}"
                )
        
        return "\n\n---\n\n".join(context_parts)
    
    def consult(self, user_input, language="english"):
        """
        Main consultation function - provides expert guidance
        
        Args:
            user_input (str): User's question or problem
            language (str): 'english' or 'bangla'
        
        Returns:
            str: Expert consultation response
        """
        if language not in ["english", "bangla"]:
            return "Error: Language must be 'english' or 'bangla'"
        
        # Analyze user intent
        intent_analysis = self.analyze_query_intent(user_input, language)
        
        # Get comprehensive context from documents
        context = self.get_enhanced_context(user_input, top_k=7)
        
        # Select appropriate prompt
        if language == "english":
            system_prompt = self.english_prompt.format(context=context)
        else:
            system_prompt = self.bangla_prompt.format(context=context)
        
        # Enhanced user message with intent
        if intent_analysis:
            user_message = f"User Query: {user_input}\n\nIntent Analysis: {intent_analysis}\n\nPlease provide comprehensive consultation and guidance."
        else:
            user_message = user_input
        
        # Generate consultative response
        try:
            response = self.openai_client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.5,  # Slightly higher for more natural consultation
                max_tokens=2500   # More tokens for comprehensive guidance
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            error_msg = {
                "english": f"I apologize, but I encountered an error while analyzing your situation: {str(e)}",
                "bangla": f"আমি দুঃখিত, আপনার পরিস্থিতি বিশ্লেষণ করার সময় একটি ত্রুটি ঘটেছে: {str(e)}"
            }
            return error_msg[language]
    
    def consult_with_details(self, user_input, language="english"):
        """
        Consultation with full details including sources and intent
        
        Returns:
            dict: Complete consultation response with metadata
        """
        # Analyze intent
        intent_analysis = self.analyze_query_intent(user_input, language)
        
        # Get documents
        results = self.search_documents(user_input, top_k=7)
        
        # Get consultation
        answer = self.consult(user_input, language)
        
        # Format sources
        sources = []
        for match in results:
            sources.append({
                'source': match.metadata.get('source', 'Unknown'),
                'page': match.metadata.get('page_number', 'N/A'),
                'relevance_score': round(match.score, 4),
                'language': match.metadata.get('language', 'unknown'),
                'content_type': match.metadata.get('content_type', 'text'),
                'text_preview': match.metadata.get('text', '')[:200]
            })
        
        return {
            'answer': answer,
            'intent_analysis': intent_analysis,
            'sources': sources,
            'total_sources_used': len(sources)
        }


def get_consultation(text_input, language="english"):
    """
    Get expert consultation on procurement issues
    
    Args:
        text_input (str): User's question or problem description
        language (str): 'english' or 'bangla'
    
    Returns:
        dict: JSON response with consultation and metadata
    """
    consultant = ProcurementConsultant(index_name="bangladesh-procurement-docs")
    result = consultant.consult_with_details(text_input, language=language)
    
    return {
        "Response": result['answer'],
    }




if __name__ == "__main__":


    answer = get_consultation(language= "english", 
                  text_input="I need to procure IT equipment worth 50 lakh taka. What method should I use?"
                  )
    
    print(answer)