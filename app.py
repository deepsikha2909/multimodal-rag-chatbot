
import streamlit as st
import hashlib
import tempfile
import os

from document_processor import extract_text

from rag_pipeline import (
    create_chunks,
    create_embeddings,
    create_vector_store,
    create_llm,
    create_rag_chain,
    ask_question
)


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# 2. CUSTOM CSS
# ==========================================

st.markdown("""
<style>

/* Main application */

.stApp {
    background-color: #F7F9FC;
}

/* Remove default top spacing */

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1150px;
}

/* Header */

.main-header {
    background: linear-gradient(120deg, #064E50, #0E7772);
    padding: 25px 30px;
    border-radius: 16px;
    margin-bottom: 25px;
    color: white;
}

.main-header h1 {
    color: white;
    font-size: 30px;
    font-weight: 700;
    margin-bottom: 5px;
}

.main-header p {
    color: #DDF4F1;
    font-size: 15px;
    margin-bottom: 0;
}

/* Sidebar */

section[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #E7EAEE;
}

.sidebar-heading {
    font-size: 20px;
    font-weight: 700;
    color: #124E50;
    margin-bottom: 5px;
}

.sidebar-description {
    font-size: 13px;
    color: #718096;
    margin-bottom: 20px;
}

/* Document card */

.document-card {
    background-color: #FFFFFF;
    border: 1px solid #DCE8E7;
    border-radius: 12px;
    padding: 16px;
    margin-top: 15px;
}

.document-name {
    color: #183B3B;
    font-size: 14px;
    font-weight: 600;
    overflow-wrap: anywhere;
}

.document-info {
    color: #718096;
    font-size: 12px;
    margin-top: 7px;
}

/* Status */

.status-success {
    background-color: #E6F7EF;
    color: #16834A;
    padding: 8px 12px;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 600;
    margin-top: 12px;
}

/* Chat section */

.chat-heading {
    font-size: 22px;
    font-weight: 700;
    color: #183B3B;
    margin-bottom: 5px;
}

.chat-subheading {
    font-size: 14px;
    color: #718096;
    margin-bottom: 20px;
}

/* Welcome screen */

.welcome-card {
    background-color: #FFFFFF;
    border: 1px solid #E5EAEF;
    border-radius: 16px;
    padding: 40px 25px;
    text-align: center;
    margin-top: 25px;
}

.welcome-icon {
    font-size: 42px;
    margin-bottom: 12px;
}

.welcome-title {
    color: #164E50;
    font-size: 23px;
    font-weight: 700;
}

.welcome-text {
    color: #718096;
    font-size: 14px;
    line-height: 1.7;
}

/* Buttons */

.stButton > button {
    background-color: #0E625E;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #084C49;
    color: white;
    border: none;
}

/* Chat input */

.stChatInput textarea {
    border-radius: 12px;
}

/* Chat message text visibility */

[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li {
    color: #183B3B !important;
}

/* Chat message headings and bold text */

[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] strong,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h1,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h2,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h3 {
    color: #124E50 !important;
}

/* Divider */

hr {
    border-color: #E7EAEE;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. HEADER
# ==========================================

st.markdown("""
<div class="main-header">
    <h1>📄 AI Document Assistant</h1>
    <p>Upload your document and get accurate answers directly
    from its content using AI.</p>
</div>
""", unsafe_allow_html=True)


# ==========================================
# 4. SIDEBAR - DOCUMENT UPLOAD
# ==========================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-heading">Document Library</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-description">'
        'Upload a document to start a conversation.'
        '</div>',
        unsafe_allow_html=True
    )

    SUPPORTED_TYPES = ["pdf", "docx", "png", "jpg", "jpeg"]

    uploaded_file = st.file_uploader(
        "Choose a document",
        type=SUPPORTED_TYPES,
        help="Supported formats: PDF, DOCX, PNG, JPG, JPEG. Maximum size: 2 MB."
    )

    st.caption("Maximum file size: 2 MB")

    if uploaded_file is None:

        st.info(
            "No document uploaded yet. Upload a file to activate the assistant."
        )


# ==========================================
# 5. PROCESS UPLOADED DOCUMENT
# ==========================================

if uploaded_file is not None:

    file_bytes = uploaded_file.getvalue()
    file_size = len(file_bytes)
    MAX_FILE_SIZE = 2 * 1024 * 1024

    if file_size > MAX_FILE_SIZE:

        st.error(
            "File size exceeds 2 MB. Please upload a smaller document."
        )

        for key in [
            "document_id",
            "rag_chain",
            "vector_store",
            "messages",
            "document_name",
            "document_size"
        ]:
            st.session_state.pop(key, None)

    else:

        file_id = hashlib.sha256(file_bytes).hexdigest()

        # Process only when a new document is uploaded

        if st.session_state.get("document_id") != file_id:

            for key in [
                "document_id",
                "rag_chain",
                "vector_store",
                "messages",
                "document_name",
                "document_size"
            ]:
                st.session_state.pop(key, None)

            file_path = None

            try:

                file_extension = os.path.splitext(
                    uploaded_file.name
                )[1].lower()

                with tempfile.NamedTemporaryFile(
                    suffix=file_extension,
                    delete=False
                ) as temp_file:

                    temp_file.write(file_bytes)
                    file_path = temp_file.name

                # Extract document text

                with st.spinner("Reading and processing your document..."):

                    document_text = extract_text(file_path)

                if not document_text.strip():

                    st.error(
                        "No readable text found. Please upload another document."
                    )

                else:

                    # Build RAG pipeline

                    with st.spinner(
                        "Creating your document knowledge base..."
                    ):

                        chunks = create_chunks(document_text)

                        if not chunks:
                            raise ValueError(
                                "No text chunks could be created."
                            )

                        embedding_model = create_embeddings()

                        vector_store = create_vector_store(
                            chunks,
                            embedding_model
                        )

                        llm = create_llm()

                        rag_chain = create_rag_chain(llm)

                    # Save session data after successful processing

                    st.session_state["vector_store"] = vector_store
                    st.session_state["rag_chain"] = rag_chain
                    st.session_state["document_id"] = file_id
                    st.session_state["messages"] = []
                    st.session_state["document_name"] = uploaded_file.name
                    st.session_state["document_size"] = file_size

                    st.rerun()

            except Exception as e:

                st.error(
                    f"Document processing failed: {e}"
                )

                for key in [
                    "document_id",
                    "rag_chain",
                    "vector_store",
                    "messages",
                    "document_name",
                    "document_size"
                ]:
                    st.session_state.pop(key, None)

            finally:

                if file_path and os.path.exists(file_path):
                    os.remove(file_path)


# ==========================================
# 6. DISPLAY UPLOADED DOCUMENT STATUS
# ==========================================

if (
    uploaded_file is not None
    and file_size <= MAX_FILE_SIZE
    and "document_id" in st.session_state
):

    with st.sidebar:

        st.markdown("---")

        st.markdown("### Uploaded Document")

        st.markdown(f"""
        <div class="document-card">
            <div class="document-name">
                📄 {st.session_state["document_name"]}
            </div>
            <div class="document-info">
                File size: {st.session_state["document_size"] / (1024 * 1024):.2f} MB
            </div>
            <div class="status-success">
                ✓ Document processed successfully
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.caption(
            "Your document is ready. You can now ask questions."
        )


# ==========================================
# 7. MAIN CHAT INTERFACE
# ==========================================

if (
    "rag_chain" in st.session_state
    and "vector_store" in st.session_state
):

    st.markdown(
        '<div class="chat-heading">Chat with your document</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="chat-subheading">'
        'Ask questions and get answers based only on your uploaded document.'
        '</div>',
        unsafe_allow_html=True
    )

    # Welcome message when conversation is empty

    if not st.session_state.get("messages"):

        st.markdown("""
        <div class="welcome-card">
            <div class="welcome-icon">💬</div>
            <div class="welcome-title">How can I help you today?</div>
            <div class="welcome-text">
                Your document has been processed successfully.<br>
                Ask a question to get started.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Display conversation history

    for message in st.session_state.get("messages", []):

        with st.chat_message(
            message["role"],
            avatar="👤" if message["role"] == "user" else "🤖"
        ):

            st.markdown(message["content"])

    # Chat input

    question = st.chat_input(
        "Ask a question about your document..."
    )

    if question:

        # Display user message

        st.session_state["messages"].append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user", avatar="👤"):
            st.markdown(question)

        # Generate AI response

        with st.chat_message("assistant", avatar="🤖"):

            with st.spinner("Analyzing your document..."):

                answer = ask_question(
                    st.session_state["vector_store"],
                    st.session_state["rag_chain"],
                    question
                )

            st.markdown(answer)

        # Save assistant response

        st.session_state["messages"].append({
            "role": "assistant",
            "content": answer
        })


# ==========================================
# 8. INITIAL SCREEN
# ==========================================

else:

    st.markdown("""
    <div class="welcome-card">
        <div class="welcome-icon">📚</div>
        <div class="welcome-title">
            Welcome to AI Document Assistant
        </div>
        <div class="welcome-text">
            Upload a PDF, Word document or image using the
            Document Library panel.<br>
            Once processing is complete, you can start asking questions.
        </div>
    </div>
    """, unsafe_allow_html=True)
