import os
import re
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("test-openai-singlefile-docs")

# Initialize OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def embed_query(query):
    """Create embedding for the search query"""
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error creating query embedding: {e}")
        return None


def search_knowledge_base(query, top_k=30, alpha=0.5):
    """Search for relevant chunks in the knowledge base"""
    query_embedding = embed_query(query)

    if not query_embedding:
        return []

    try:
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            alpha=alpha
        )

        relevant_chunks = []
        for match in results.matches:
            if match.score > 0.2:  # Lower threshold for better recall
                relevant_chunks.append({
                    'text': match.metadata.get('text', ''),
                    'source': match.metadata.get('source', 'Unknown'),
                    'page': match.metadata.get('page', 'N/A'),
                    'chunk': match.metadata.get('chunk', None),
                    'total_chunks': match.metadata.get('total_chunks', None),
                    'score': match.score
                })

        relevant_chunks.sort(key=lambda x: x['score'], reverse=True)
        return relevant_chunks

    except Exception as e:
        print(f"Search error: {e}")
        return []


def group_chunks_by_context(chunks, max_context_length=10000):
    """Group chunks by source and page to maintain context"""
    if not chunks:
        return []

    # Group by source and page
    page_groups = {}
    for chunk in chunks:
        key = (chunk['source'], chunk['page'])
        if key not in page_groups:
            page_groups[key] = []
        page_groups[key].append(chunk)

    # Sort chunks within each page by chunk number
    for key in page_groups:
        page_groups[key].sort(key=lambda x: x.get('chunk', 0) if x.get('chunk') else 0)

    # Create context groups
    context_groups = []
    current_group = []
    current_length = 0

    # Sort pages by highest relevance score
    sorted_pages = sorted(
        page_groups.keys(),
        key=lambda k: max(chunk['score'] for chunk in page_groups[k]),
        reverse=True
    )

    for page_key in sorted_pages:
        page_chunks = page_groups[page_key]
        page_text = " ".join([chunk['text'] for chunk in page_chunks])

        if current_length + len(page_text) <= max_context_length:
            current_group.extend(page_chunks)
            current_length += len(page_text)
        else:
            if current_group:
                context_groups.append(current_group)
            current_group = page_chunks[:]
            current_length = len(page_text)

    if current_group:
        context_groups.append(current_group)

    return context_groups


def answer_question_with_context(query, context_groups, language):
    """Generate answer using OpenAI with provided context"""
    if not context_groups:
        if language == "bangla":
            return "দুঃখিত, প্রশ্নের সাথে সম্পর্কিত পর্যাপ্ত তথ্য পাওয়া যায়নি।", []
        return "Not enough relevant information found.", []

    all_contexts = []
    sources_info = []

    for i, group in enumerate(context_groups):
        group_text = ""
        group_sources = set()
        group_pages = set()

        for chunk in group:
            group_text += chunk['text'] + " "
            group_sources.add(chunk['source'])
            group_pages.add(str(chunk['page']))

        context_header = f"Context {i+1} (Source: {', '.join(group_sources)}, Pages: {', '.join(sorted(group_pages))}):\n"
        all_contexts.append(context_header + group_text.strip())
        sources_info.extend([(chunk['source'], chunk['page'], chunk['score']) for chunk in group])

    full_context = "\n\n".join(all_contexts)

    # Language-specific prompts
    if language == "bangla":
        prompt = f"""আপনি বাংলাদেশ সরকারি ক্রয় বিধিমালা ২০২৫ (PPR 2025) বিধি, নিয়মকানুন এবং পদ্ধতির একজন বিশেষজ্ঞ পরামর্শদাতা। আপনার দায়িত্ব হলো সংগ্রহ সংক্রান্ত বিষয়ে স্পষ্ট, ব্যবহারিক এবং নির্ভুল পরামর্শ প্রদান করা।

        নির্দেশনা:
        1. সমস্ত প্রদত্ত প্রেক্ষাপট (কনটেক্সট) মনোযোগসহ বিশ্লেষণ করুন এবং নিয়ম ও প্রেক্ষাপট বুঝে সর্বপ্রথম সঠিক, সংক্ষিপ্ত ও সরাসরি উত্তর বা সমাধান দিন; প্রয়োজনে পরে ব্যাখ্যা করুন। (অনুমান করবেন না)
        2. প্রাসঙ্গিক হলে একাধিক প্রসঙ্গ থেকে তথ্য ব্যবহার করুন
        3. শুধুমাত্র প্রদত্ত প্রসঙ্গের উপর ভিত্তি করে একটি বিস্তৃত উত্তর প্রদান করুন, এবং উত্সগুলির রেফারেন্স প্রদান করুন(পৃষ্ঠা নম্বর , নিয়ম নম্বর, বিভাগ নম্বর ইত্যাদি প্রদান করুন যদি সম্ভব হয়)
        4. প্রসঙ্গ এবং এটি কী বোঝায় তা বুঝুন এবং সর্বোত্তম সম্ভাব্য উত্তর প্রদান করুন
        5. প্রসঙ্গে পর্যাপ্ত তথ্য না থাকলে, এটি স্পষ্টভাবে বলুন
        6. আপনার প্রতিক্রিয়ায় সুনির্দিষ্ট এবং বিস্তারিত হন
        7. সবসময় বাংলায় উত্তর দিন
        8. সঠিক মার্কডাউন ফরম্যাটে উত্তর দিন

        নথি থেকে প্রসঙ্গ:
        {full_context}

        প্রশ্ন: {query}

        উত্তর:"""
    else:
        prompt = f"""You are an expert consultant specializing in Bangladesh Public Procurement Rules 2025 (PPR 2025) regulations, rules, and procedures. Your role is to provide clear, practical, and accurate guidance on procurement-related matters.

        INSTRUCTIONS:
        1. Analyze all the provided contexts carefully and give the accurate short direct answer or solution first by understanding the rules and context then explain if needed. (don't spaculate)
        2. Use information from multiple contexts when relevant
        3. Provide a comprehensive answer based only on the given contexts, and provide references to the sources(Porvide page numbers, rules no. section no. etc. if possible)
        4. Understand the context and what it implies to provide the best possible answer
        5. If contexts don't contain enough information, say so clearly
        6. Be specific and detailed in your response
        7. Always respond in English
        8 Answer in proper marksdown format

        CONTEXTS FROM DOCUMENT:
        {full_context}

        QUESTION: {query}

        ANSWER:"""

    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are an expert consultant on Bangladesh Public Procurement Rules 2025."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5
        )
        return response.choices[0].message.content, sources_info
    except Exception as e:
        print(f"Error generating answer: {e}")
        if language == "bangla":
            return f"দুঃখিত, উত্তর তৈরিতে সমস্যা হয়েছে: {str(e)}", []
        return f"Sorry, having some issue with generating answer: {str(e)}", []


def get_answer(query, language):
    """Main function to get answer for a query"""
    
    print(f"Using language: {language}")

    # Search for relevant chunks
    relevant_chunks = search_knowledge_base(query, top_k=40)

    if not relevant_chunks:
        if language == "bangla":
            return "দুঃখিত, প্রশ্নের সাথে সম্পর্কিত কোনো তথ্য পাওয়া যায়নি।", []
        return "No relevant answer found for the given question.", []

    print(f"Found {len(relevant_chunks)} relevant chunks")

    # Group chunks by context
    context_groups = group_chunks_by_context(relevant_chunks, max_context_length=10000)
    
    print(f"Grouped into {len(context_groups)} context groups")

    # Generate answer
    answer, sources = answer_question_with_context(query, context_groups, language)

    return answer, sources


def get_consultation(text_input, language):
    """Wrapper function for easy use"""
    answer, sources = get_answer(text_input, language)
    
    return {
        "answer": answer,
        "language": language,
        # Uncomment to see sources
        # "sources": [{"source": s[0], "page": s[1], "score": round(s[2], 4)} for s in sources[:10]]
    }


# Example Usage:
if __name__ == "__main__":

    result = get_consultation(
        text_input="what is call off in ppr 2025?",
        language="english"
    )
    print(f"\nAnswer:\n{result['answer']}")