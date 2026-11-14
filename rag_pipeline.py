
import time
import re
import uuid

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from config import API_KEY, MODEL_NAME


REFUSAL_MESSAGE = "Sorry I could not find any information"

RETRIEVAL_COUNT = 6


# ==========================================
# STEP 1: TEXT CHUNKING
# ==========================================

def create_chunks(text):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""],
        add_start_index=True
    )

    chunks = text_splitter.create_documents([text])

    print("\nTotal chunks created:", len(chunks))

    return chunks


# ==========================================
# STEP 2: CREATE EMBEDDINGS
# ==========================================

def create_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# ==========================================
# STEP 3: CREATE VECTOR STORE
# ==========================================

def create_vector_store(chunks, embedding_model):

    # Create a unique collection for each uploaded document.
    # This prevents different documents from sharing a collection.

    collection_name = "rag_" + uuid.uuid4().hex

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name=collection_name
    )

    print("\nVector store created:", collection_name)

    print(
        "Stored document chunks:",
        vector_store._collection.count()
    )

    return vector_store


# ==========================================
# STEP 4: CREATE RETRIEVER
# ==========================================

def create_retriever(vector_store):

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVAL_COUNT}
    )


# ==========================================
# STEP 5: GET ALL DOCUMENT CHUNKS
# ==========================================

def get_all_document_chunks(vector_store):

    stored_data = vector_store.get(
        include=["documents", "metadatas"]
    )

    documents = []

    for content, metadata in zip(
        stored_data["documents"],
        stored_data["metadatas"]
    ):

        documents.append(
            Document(
                page_content=content,
                metadata=metadata or {}
            )
        )

    documents.sort(
        key=lambda doc: doc.metadata.get(
            "start_index", 0
        )
    )

    return documents


# ==========================================
# STEP 6: EXTRACT KEYWORDS
# ==========================================

def extract_keywords(text):

    words = re.findall(
        r"[a-zA-Z0-9]+",
        text.lower()
    )

    stopwords = {
        "the", "is", "are", "was", "were",
        "a", "an", "of", "in", "on", "at",
        "to", "for", "from", "and", "or",
        "what", "which", "who", "how",
        "give", "me", "tell", "about",
        "please", "can", "you", "do",
        "does", "this", "that", "it",
        "my", "your", "there", "with",
        "show", "find", "provide"
    }

    return [
        word for word in words
        if word not in stopwords
    ]


# ==========================================
# STEP 7: CALCULATE KEYWORD SCORE
# ==========================================

def calculate_keyword_score(question, document_text):

    question_keywords = set(
        extract_keywords(question)
    )

    document_keywords = set(
        extract_keywords(document_text)
    )

    if not question_keywords:
        return 0

    matching_keywords = (
        question_keywords.intersection(
            document_keywords
        )
    )

    keyword_score = (
        len(matching_keywords) /
        len(question_keywords)
    )

    # Give additional weight to exact phrase matches.

    normalized_question = " ".join(
        extract_keywords(question)
    )

    normalized_document = " ".join(
        extract_keywords(document_text)
    )

    if (
        normalized_question
        and normalized_question in normalized_document
    ):

        keyword_score += 0.5

    return keyword_score


# ==========================================
# STEP 8: DOCUMENT HEADER RETRIEVAL
# ==========================================

def retrieve_header(documents, question):

    question_lower = question.lower()

    header_keywords = [
        "subject code",
        "course code",
        "subject name",
        "course name",
        "college name",
        "institution name",
        "unit number",
        "which unit",
        "which units",
        "department name",
        "university name",
        "institution"
    ]

    if not any(
        keyword in question_lower
        for keyword in header_keywords
    ):

        return None

    print("\nSPECIAL RETRIEVAL: DOCUMENT HEADER")

    header_documents = [
        doc for doc in documents
        if doc.metadata.get("start_index", 0) < 600
    ]

    if header_documents:

        return [
            (doc, 1.0)
            for doc in header_documents[:3]
        ]

    return None


# ==========================================
# STEP 9: NUMBERED QUESTION RETRIEVAL
# ==========================================

def identify_big_question_position(documents):

    for doc in documents:

        match = re.search(
            r"\bBig\s+Question\b",
            doc.page_content,
            re.IGNORECASE
        )

        if match:

            start_index = doc.metadata.get(
                "start_index", 0
            )

            return start_index + match.start()

    return None


def retrieve_numbered_question(documents, question):

    question_lower = question.lower()

    number_match = re.search(
        r"\b(\d+)(?:st|nd|rd|th)?\s+question\b",
        question_lower
    )

    if not number_match:
        return None

    question_number = int(
        number_match.group(1)
    )

    big_question_position = identify_big_question_position(
        documents
    )

    requested_section = None

    if re.search(
        r"big\s+question|big\s+questions",
        question_lower
    ):

        requested_section = "big"

    elif re.search(
        r"section\s*[- ]?1|main\s+question",
        question_lower
    ):

        requested_section = "main"

    print(
        "\nSPECIAL RETRIEVAL: QUESTION NUMBER",
        question_number
    )

    matching_documents = []

    for doc in documents:

        content = doc.page_content

        start_index = doc.metadata.get(
            "start_index", 0
        )

        for match in re.finditer(
            rf"(?m)^\s*{question_number}\s*[\.\)]\s*",
            content
        ):

            absolute_position = (
                start_index + match.start()
            )

            is_big_question = (
                big_question_position is not None
                and absolute_position >= big_question_position
            )

            if (
                requested_section == "main"
                and is_big_question
            ):
                continue

            if (
                requested_section == "big"
                and not is_big_question
            ):
                continue

            matching_documents.append(doc)
            break

    # Remove duplicate chunks.

    unique_documents = []
    seen = set()

    for doc in matching_documents:

        key = doc.metadata.get(
            "start_index",
            doc.page_content
        )

        if key not in seen:

            seen.add(key)
            unique_documents.append(doc)

    if unique_documents:

        return [
            (doc, 1.0)
            for doc in unique_documents[:3]
        ]

    return None


# ==========================================
# STEP 10: KEYWORD RETRIEVAL
# ==========================================

def keyword_retrieval(documents, question):

    results = []

    for doc in documents:

        score = calculate_keyword_score(
            question,
            doc.page_content
        )

        if score > 0:

            results.append(
                (doc, score)
            )

    results.sort(
        key=lambda item: item[1],
        reverse=True
    )

    return results


# ==========================================
# STEP 11: HYBRID RETRIEVAL
# ==========================================

def retrieve_documents(vector_store, question):

    print("\n========== HYBRID RETRIEVAL ==========")
    print("Question:", question, flush=True)

    # Retrieve all chunks belonging to this document.

    all_documents = get_all_document_chunks(
        vector_store
    )

    if not all_documents:

        print("No document chunks available.")

        return []

    # --------------------------------------
    # A. NUMBERED QUESTION RETRIEVAL
    # --------------------------------------

    numbered_results = retrieve_numbered_question(
        all_documents,
        question
    )

    if numbered_results:

        print("Retrieval method: Numbered question")

        return display_retrieved_results(
            numbered_results
        )

    # --------------------------------------
    # B. HEADER RETRIEVAL
    # --------------------------------------

    header_results = retrieve_header(
        all_documents,
        question
    )

    if header_results:

        print("Retrieval method: Document header")

        return display_retrieved_results(
            header_results
        )

    # --------------------------------------
    # C. SEMANTIC RETRIEVAL
    # --------------------------------------

    print("\nRunning ChromaDB semantic retrieval...")

    semantic_results = (
        vector_store.similarity_search_with_relevance_scores(
            question,
            k=RETRIEVAL_COUNT
        )
    )

    # --------------------------------------
    # D. KEYWORD RETRIEVAL
    # --------------------------------------

    print("\nRunning keyword retrieval...")

    keyword_results = keyword_retrieval(
        all_documents,
        question
    )

    # --------------------------------------
    # E. COMBINE SEMANTIC AND KEYWORD RESULTS
    # --------------------------------------

    combined = {}

    # Add semantic results.

    for rank, (doc, score) in enumerate(
        semantic_results
    ):

        key = doc.metadata.get(
            "start_index",
            doc.page_content
        )

        combined[key] = {
            "document": doc,
            "semantic_rank": rank,
            "keyword_score": 0
        }

    # Add keyword results.

    for doc, keyword_score in keyword_results:

        key = doc.metadata.get(
            "start_index",
            doc.page_content
        )

        if key in combined:

            combined[key]["keyword_score"] = keyword_score

        else:

            combined[key] = {
                "document": doc,
                "semantic_rank": RETRIEVAL_COUNT + 1,
                "keyword_score": keyword_score
            }

    # Rank documents using both retrieval methods.

    ranked_results = sorted(
        combined.values(),
        key=lambda item: (
            item["keyword_score"],
            -item["semantic_rank"]
        ),
        reverse=True
    )

    results = []

    for item in ranked_results[:RETRIEVAL_COUNT]:

        results.append(
            (
                item["document"],
                item["keyword_score"]
            )
        )

    print("\nRetrieval method: Hybrid retrieval")

    return display_retrieved_results(results)


# ==========================================
# STEP 12: DISPLAY RETRIEVED RESULTS
# ==========================================

def display_retrieved_results(results):

    print("\n========== RETRIEVED DOCUMENT CHUNKS ==========")

    if not results:

        print("No matching chunks found.")

        return []

    for i, (doc, score) in enumerate(
        results,
        start=1
    ):

        print(f"\nCHUNK {i}")

        print(
            "Retrieval score:",
            round(float(score), 3)
        )

        print("Document content:")
        print(doc.page_content)

    print("\n================================================")

    return results


# ==========================================
# STEP 13: GEMINI LLM
# ==========================================

def create_llm():

    return ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=API_KEY
    )


# ==========================================
# STEP 14: CREATE RAG CHAIN
# ==========================================

def create_rag_chain(llm):

    system_prompt = """
    You are a document-based AI assistant.

    Answer questions using ONLY the supplied document context.

    STRICT RULES:

    1. Use only information explicitly supported by the context.

    2. Never use outside knowledge.

    3. Never invent subject codes, course codes,
       names, dates, numbers or other details.

    4. When asked for a subject code, return the code
       explicitly found in the context.

    5. When asked for a subject name, return the name
       explicitly found in the context.

    6. When asked for a unit number, return the unit
       information explicitly found in the context.

    7. When asked for a numbered question, return the
       matching question from the supplied context.

    8. Respect the requested section when the user
       specifies a section.

    9. When asked about dashboard statistics, carefully
       associate each number with its corresponding label.

    10. For example, if the context states:
        Total Students: 3
        Total Male Students: 1
        Total Female Students: 2

        Then answer each question using its matching value.

    11. Do not confuse numbers belonging to different labels.

    12. Combine multiple supplied chunks when necessary.

    13. If the requested answer cannot be found in the
        supplied context, respond EXACTLY:

        Sorry I could not find any information

    14. Do not guess or provide alternative answers.

    DOCUMENT CONTEXT:

    {context}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])

    return prompt | llm


# ==========================================
# STEP 15: EXTRACT GEMINI RESPONSE
# ==========================================

def extract_response_text(response):

    if isinstance(response.content, str):

        return response.content.strip()

    text_parts = []

    for item in response.content:

        if isinstance(item, dict):

            if item.get("type") == "text":

                text_parts.append(
                    item.get("text", "")
                )

        elif isinstance(item, str):

            text_parts.append(item)

    return "\n".join(text_parts).strip()


# ==========================================
# STEP 16: PERFORMANCE REPORT
# ==========================================

def print_performance_report(
    retrieval_time,
    gemini_time,
    total_time
):

    print("\n========== PERFORMANCE REPORT ==========")

    print(
        "Retrieval time:",
        round(retrieval_time, 3),
        "seconds"
    )

    print(
        "Gemini response time:",
        round(gemini_time, 3),
        "seconds"
    )

    print(
        "Total response time:",
        round(total_time, 3),
        "seconds"
    )

    print("=========================================")


# ==========================================
# STEP 17: ASK QUESTION
# ==========================================

def ask_question(
    vector_store,
    rag_chain,
    question
):

    total_start = time.perf_counter()

    print("\n========== USER QUESTION ==========")
    print(question, flush=True)

    # --------------------------------------
    # DOCUMENT RETRIEVAL
    # --------------------------------------

    retrieval_start = time.perf_counter()

    results = retrieve_documents(
        vector_store,
        question
    )

    retrieval_time = (
        time.perf_counter() - retrieval_start
    )

    if not results:

        total_time = (
            time.perf_counter() - total_start
        )

        print_performance_report(
            retrieval_time,
            0,
            total_time
        )

        return REFUSAL_MESSAGE

    # --------------------------------------
    # PREPARE CONTEXT
    # --------------------------------------

    context = "\n\n".join(
        doc.page_content
        for doc, score in results
    )

    print("\n========== CONTEXT SENT TO GEMINI ==========")
    print(context)
    print("============================================")

    # --------------------------------------
    # GEMINI RESPONSE
    # --------------------------------------

    try:

        print("\nSending question to Gemini...")

        gemini_start = time.perf_counter()

        response = rag_chain.invoke({
            "context": context,
            "question": question
        })

        gemini_time = (
            time.perf_counter() - gemini_start
        )

        answer = extract_response_text(response)

        if not answer:

            answer = REFUSAL_MESSAGE

        total_time = (
            time.perf_counter() - total_start
        )

        print_performance_report(
            retrieval_time,
            gemini_time,
            total_time
        )

        print("\n========== GEMINI ANSWER ==========")
        print(answer)
        print("===================================")

        return answer

    except Exception as e:

        total_time = (
            time.perf_counter() - total_start
        )

        print("\nGemini API error:", str(e))

        print_performance_report(
            retrieval_time,
            0,
            total_time
        )

        return (
            "An error occurred while generating the answer. "
            "Please try again."
        )


# ==========================================
# STEP 18: MAIN TESTING
# ==========================================

if __name__ == "__main__":

    sample_text = """
    Artificial Intelligence is transforming many industries.

    Machine Learning is a subset of Artificial Intelligence.

    Deep Learning uses neural networks to learn complex patterns.
    """

    # Create chunks

    chunks = create_chunks(sample_text)

    # Load embeddings

    print("\nLoading Hugging Face embedding model...")

    embedding_model = create_embeddings()

    vector = embedding_model.embed_query(
        chunks[0].page_content
    )

    print("\nEmbedding generated successfully!")

    print("Vector dimensions:", len(vector))

    # Create vector store

    print("\nCreating ChromaDB vector store...")

    vector_store = create_vector_store(
        chunks,
        embedding_model
    )

    # Connect Gemini

    print("\nConnecting to Gemini...")

    llm = create_llm()

    # Create RAG chain

    print("\nCreating RAG chain...")

    rag_chain = create_rag_chain(llm)

    # Test questions

    questions = [
        "What is Machine Learning?",
        "Who won the cricket World Cup?"
    ]

    for question in questions:

        answer = ask_question(
            vector_store,
            rag_chain,
            question
        )

        print("\nFINAL ANSWER:")
        print(answer)

        print("\n===================================")
