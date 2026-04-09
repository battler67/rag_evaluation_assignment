from dotenv import load_dotenv
import streamlit as st
import os
import json
import pandas as pd
from rag_pipeline import TardigradeRAG
from evaluation import RAGEvaluator

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Tardigrade RAG App", layout="wide")

# Simple, professional UI without emojis as requested
st.title("Tardigrade RAG System & Evaluation")
st.markdown("A localized Retrieval-Augmented Generation system focused on Tardigrade biology.")

# Initialize Session State
if "rag" not in st.session_state:
    st.session_state.rag = None
if "eval_results" not in st.session_state:
    st.session_state.eval_results = None

# Sidebar for Setup and Configuration
st.sidebar.header("System Setup")
# api_key = st.sidebar.text_input("OpenAI API Key (or check .env)", value=os.environ.get("OPENAI_API_KEY", ""), type="password")
api_key = os.environ.get("OPENAI_API_KEY", "")
if st.sidebar.button("Initialize Pipeline"):
    if not api_key:
        st.sidebar.error("OpenAI API Key is required to run the Generator.")
    else:
        os.environ["OPENAI_API_KEY"] = api_key
        with st.spinner("Initializing RAG Pipeline and Vector Store..."):
            rag = TardigradeRAG()
            rag.build_vector_store()
            rag.setup_rag_chain()
            st.session_state.rag = rag
        st.sidebar.success("Pipeline Initialized successfully.")

st.sidebar.markdown("---")
st.sidebar.header("Evaluation")
if st.sidebar.button("Run Quantitative Evaluation"):
    if st.session_state.rag is None:
        st.sidebar.error("Please initialize the system first.")
    else:
        with st.spinner("Running evaluation over dataset..."):
            evaluator = RAGEvaluator()
            results = evaluator.evaluate_all("data/qa_dataset.json", "data/eval_results.json", st.session_state.rag)
            st.session_state.eval_results = results
        st.sidebar.success("Evaluation complete.")

tab1, tab2 = st.tabs(["Chat & Query", "Evaluation Results"])

with tab1:
    st.subheader("Query the Tardigrade Knowledge Base")
    user_query = st.text_input("Ask a question about tardigrades:")
    
    if st.button("Search"):
        if st.session_state.rag is None:
            st.error("System is not initialized. Please configure the sidebar first.")
        elif not user_query:
            st.warning("Please enter a question.")
        else:
            with st.spinner("Retrieving and generating answer..."):
                try:
                    response = st.session_state.rag.query(user_query)
                    st.markdown("### Answer")
                    st.write(response["answer"])
                    
                    st.markdown("### Retrieved Context")
                    for i, doc in enumerate(response["source_documents"]):
                        with st.expander(f"Source Chunk {i+1}"):
                            st.write(doc.page_content)
                except Exception as e:
                    st.error(f"Error during query: {str(e)}")

with tab2:
    st.subheader("Evaluation Framework Dashboard")
    st.markdown("This tab displays the results of the quantitative evaluation running against `data/qa_dataset.json`.")
    
    if st.session_state.eval_results is None:
        st.info("Run the evaluation from the sidebar to view results.")
    else:
        results = st.session_state.eval_results
        
        # Prepare dataframe for display
        df_data = []
        for r in results:
            if "error" in r:
                continue
            df_data.append({
                "Question": r["question"],
                "Semantic Similarity": r["quantitative_metrics"]["cosine_similarity"],
                "Keyword Overlap": r["quantitative_metrics"]["keyword_overlap"],
                "Retrieval Score": r.get("retrieval_performance", {}).get("keyword_presence_in_context", 0.0)
            })
            
        if df_data:
            df = pd.DataFrame(df_data)
            st.dataframe(df.style.format({
                "Semantic Similarity": "{:.2f}",
                "Keyword Overlap": "{:.2f}",
                "Retrieval Score": "{:.2f}"
            }), use_container_width=True)
            
            st.markdown("### Detailed Qualitative Grading Sheet")
            st.markdown("Use this section to perform human qualitative assessment on the generated outputs.")
            for i, r in enumerate(results):
                if "error" in r:
                    continue
                with st.expander(f"Q: {r['question']}"):
                    st.markdown(f"**Expected Answer:** {r['expected_answer']}")
                    st.markdown(f"**Generated Answer:** {r['generated_answer']}")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.slider(f"Coherence (1-5) - {i}", 1, 5, 3, key=f"coh_{i}")
                    with col2:
                        st.slider(f"Factual Correctness (1-5) - {i}", 1, 5, 3, key=f"fac_{i}")
                    with col3:
                        st.slider(f"Completeness (1-5) - {i}", 1, 5, 3, key=f"com_{i}")
                    st.text_input(f"Notes - {i}", key=f"note_{i}")
            #this is ne