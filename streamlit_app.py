"""
Simplified Streamlit App for Ethical AI RAG System
Deployment-ready version with minimal dependencies
"""

import sys

try:
    import PyPDF2
except ImportError:
    import streamlit as st
    st.error("❌ PyPDF2 is not installed. Please run: pip install -r requirements.txt")
    import sys
    sys.exit(1)

import streamlit as st
import os
import logging
import tempfile
from pathlib import Path
from typing import Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Environment setup
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

# Session state initialization
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'system_initialized' not in st.session_state:
    st.session_state.system_initialized = False
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False

def safe_import_rag():
    """Safely import RAG system with error handling."""
    try:
        # Only import when actually needed
        from rag_system import AdvancedRAGSystem
        return AdvancedRAGSystem, None
    except ImportError as e:
        return None, f"Import error: {e}"
    except Exception as e:
        return None, f"Unexpected error: {e}"

def initialize_rag_system():
    """Initialize the RAG system with error handling."""
    if st.session_state.system_initialized and st.session_state.rag_system:
        return st.session_state.rag_system, None
    
    try:
        RAGSystem, error = safe_import_rag()
        if error:
            return None, error
        
        if RAGSystem is None:
            return None, "Failed to import RAG system class"
        
        # Initialize with minimal configuration - NO MODEL LOADING
        rag_system = RAGSystem(
            pdf_directory=".",
            vector_db_path="./vector_db",
            embedding_model="tfidf",
            llm_provider="enhanced_simple",  # Use simple generation to avoid complex dependencies
            llm_model="simple"
        )
        
        # Don't initialize LLM during startup
        rag_system.llm_pipeline = None
        rag_system.tokenizer = None
        
        st.session_state.rag_system = rag_system
        st.session_state.system_initialized = True
        logger.info("✅ RAG system initialized successfully (minimal mode)")
        return rag_system, None
        
    except Exception as e:
        error_msg = f"Failed to initialize RAG system: {e}"
        logger.error(error_msg)
        return None, error_msg

def process_uploaded_files(uploaded_files) -> bool:
    """Process multiple uploaded PDF files."""
    try:
        rag_system, error = initialize_rag_system()
        if error:
            st.error(f"❌ {error}")
            return False
        
        if rag_system is None:
            st.error("❌ RAG system is not initialized")
            return False
        
        total_documents = 0
        successful_files = 0
        
        # Process each file
        for uploaded_file in uploaded_files:
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                try:
                    # Extract text
                    text = rag_system.pdf_extractor.extract_text_robust(tmp_file_path)
                    if not text or len(text.strip()) < 50:
                        st.warning(f"⚠️ Could not extract meaningful text from {uploaded_file.name}")
                        continue
                    
                    # Process into documents
                    temp_doc = {uploaded_file.name: text}
                    rag_documents = rag_system.document_processor.create_rag_documents(temp_doc)
                    
                    if not rag_documents:
                        st.warning(f"⚠️ Could not process text into chunks for {uploaded_file.name}")
                        continue
                    
                    # Add to vector store
                    success = rag_system.vector_store.create_vectorstore(rag_documents)
                    if success:
                        total_documents += len(rag_documents)
                        successful_files += 1
                        st.success(f"✅ Successfully processed {uploaded_file.name} into {len(rag_documents)} chunks")
                    else:
                        st.error(f"❌ Failed to add {uploaded_file.name} to vector store")
                        
                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(tmp_file_path)
                    except:
                        pass
                        
            except Exception as e:
                st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                logger.error(f"File processing error for {uploaded_file.name}: {e}")
        
        # Update system state if any files were processed successfully
        if successful_files > 0:
            rag_system.is_ready = True
            st.session_state.documents_processed = True
            st.success(f"✅ Successfully processed {successful_files} files with {total_documents} total chunks")
            return True
        else:
            st.error("❌ No files were processed successfully")
            return False
        
    except Exception as e:
        st.error(f"❌ Error processing files: {str(e)}")
        logger.error(f"File processing error: {e}")
        return False

def ask_question(question: str) -> Dict[str, Any]:
    """Ask a question using the RAG system."""
    try:
        rag_system, error = initialize_rag_system()
        if error:
            return {"error": error}
        
        if rag_system is None:
            return {"error": "RAG system is not initialized"}
        
        if not rag_system.is_ready or not st.session_state.documents_processed:
            return {"error": "No documents processed yet. Please upload PDF files first."}
        
        # Get answer from RAG system
        result = rag_system.ask_question(question)
        return result
        
    except Exception as e:
        logger.error(f"Question processing error: {e}")
        return {"error": f"Failed to process question: {str(e)}"}

def main():
    """Main Streamlit application."""
    
    # Page configuration
    st.set_page_config(
        page_title="Ethical AI RAG System",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Add custom CSS for stable layout
    st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    .stColumns > div {
        min-height: 400px;
    }
    
    .stTextArea textarea {
        min-height: 100px !important;
    }
    
    .upload-section, .question-section {
        height: 100%;
        display: flex;
        flex-direction: column;
    }
    
    .stExpander {
        max-height: 200px;
        overflow-y: auto;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("🤖 Ethical AI RAG System")
    st.markdown("**Document Analysis with AI-Powered Question Answering**")
    
    # System info
    st.info("""
    🚀 **Features:**
    - ✅ Multiple PDF document processing
    - ✅ TF-IDF embeddings for semantic search
    - ✅ AI-powered question answering
    - ✅ Document chunking and analysis
    - ✅ Deployment-ready architecture
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("📊 System Status")
        
        # Initialize system button
        if st.button("🔄 Initialize System"):
            rag_system, error = initialize_rag_system()
            if error:
                st.error(f"❌ {error}")
            else:
                st.success("✅ System initialized successfully!")
        
        # Status display
        if st.session_state.system_initialized:
            st.success("✅ System Ready")
            if st.session_state.documents_processed:
                st.success("✅ Documents Processed")
            else:
                st.warning("⚠️ No documents processed")
        else:
            st.warning("⚠️ System not initialized")
        
        st.header("🔧 Controls")
        if st.button("🗑️ Clear All Data"):
            st.session_state.rag_system = None
            st.session_state.system_initialized = False
            st.session_state.documents_processed = False
            st.success("✅ All data cleared")
            st.rerun()
    
    # Main content
    # Create a container for the main content area
    main_container = st.container()
    
    with main_container:
        # Use columns with equal width and better height management
        col1, col2 = st.columns([1, 1], gap="medium")
        
        with col1:
            # Create a container for the upload section with fixed height
            st.markdown('<div class="upload-section">', unsafe_allow_html=True)
            upload_container = st.container()
            with upload_container:
                st.header("📄 Document Upload")
                st.markdown("Upload PDF documents for analysis")
                
                uploaded_files = st.file_uploader(
                    "Choose PDF files",
                    type="pdf",
                    help="Upload multiple PDF documents to analyze with the RAG system",
                    accept_multiple_files=True
                )
                
                if uploaded_files:
                    st.write(f"**Total Files:** {len(uploaded_files)}")
                    st.write(f"**Total Size:** {sum(f.size for f in uploaded_files):,} bytes")
                    
                    # Display file list in a scrollable container
                    with st.expander("📋 View Uploaded Files", expanded=False):
                        for i, file in enumerate(uploaded_files, 1):
                            st.write(f"**{i}.** {file.name} ({file.size:,} bytes)")
                    
                    if st.button("🔄 Process Documents"):
                        with st.spinner("Processing documents..."):
                            success = process_uploaded_files(uploaded_files)
                            if success:
                                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            # Create a container for the question section with fixed height
            st.markdown('<div class="question-section">', unsafe_allow_html=True)
            question_container = st.container()
            with question_container:
                st.header("❓ Ask Questions")
                st.markdown("Query your documents using AI analysis")
                
                question = st.text_area(
                    "Enter your question:",
                    placeholder="e.g., What are the main ethical principles for AI development?",
                    height=100
                )
                
                if st.button("🔍 Get Answer") and question.strip():
                    with st.spinner("Generating answer..."):
                        result = ask_question(question.strip())
                        
                        if "error" in result:
                            st.error(f"❌ {result['error']}")
                        else:
                            st.success("✅ Answer generated successfully!")
                            
                            # Display answer
                            st.subheader("📋 Answer")
                            st.write(result.get("answer", "No answer generated"))
                            
                            # Display metadata
                            if result.get('confidence'):
                                st.metric("Confidence", f"{result['confidence']:.1%}")
                            
                            # Display sources
                            if result.get('sources'):
                                st.subheader("📚 Sources")
                                for i, source in enumerate(result['sources'], 1):
                                    with st.expander(f"Source {i}"):
                                        st.write(source.get('content', 'No content'))
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Ethical AI RAG System** - Powered by TF-IDF and AI
    
    🔗 **How it works:**
    1. Upload multiple PDF documents
    2. System processes and chunks the text
    3. Ask questions about the content
    4. Get AI-powered answers with source references
    """)

if __name__ == "__main__":
    main() 