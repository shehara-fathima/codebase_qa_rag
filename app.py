import os
import shutil
import gc
import time
from dotenv import load_dotenv
import streamlit as st
from git import Repo
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import google.generativeai as genai
from typing import List, Dict, Tuple
from main import CodebaseQA, setup_gemini, clear_repository_data

def main():
    st.set_page_config(page_title="Codebase QA Assistant", page_icon="💻")
    st.title("Codebase QA Assistant")

    setup_gemini()

    if 'qa_system' not in st.session_state:
        st.session_state['qa_system'] = CodebaseQA()

    with st.sidebar:
        st.header("Configuration")
        repo_url = st.text_input("Git Repository URL",
                                 placeholder="https://github.com/user/repo.git",
                                 key="repo_url_input")

        if st.button("Load Repository"):
            if repo_url:
                with st.spinner("Processing repository..."):
                    try:
                        clear_repository_data()
                        st.session_state['qa_system'] = CodebaseQA()
                        repo_path = st.session_state['qa_system'].clone_repository(repo_url)
                        docs = st.session_state['qa_system'].load_files(repo_path)
                        embeddings, chunks, metas = st.session_state['qa_system'].embed_documents(docs)
                        st.session_state['qa_system'].index = st.session_state['qa_system'].create_index(embeddings)
                        st.session_state['qa_system'].chunks = chunks
                        st.session_state['qa_system'].metas = metas
                        st.session_state['current_repo'] = repo_url
                        st.success(f"Processed {len(docs)} files with {len(chunks)} chunks!")
                    except Exception as e:
                        st.error(f"Error processing repository: {str(e)}")
            else:
                st.warning("Please enter a repository URL")

    if 'current_repo' in st.session_state:
        st.subheader(f"Current Repository: {st.session_state['current_repo']}")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Ask New Question on Same Repository"):
                if 'last_query' in st.session_state:
                    del st.session_state['last_query']
                st.rerun()

        with col2:
            if st.button("Load New Repository"):
                clear_repository_data()
                st.rerun()

        st.subheader("Ask a Question")
        query = st.text_input("Your question about the codebase",
                              placeholder="What does this function do?",
                              key="query_input")

        if query:
            st.session_state['last_query'] = query
            with st.spinner("Searching for answers..."):
                results = st.session_state['qa_system'].search_index(query)
                answer = st.session_state['qa_system'].ask_gemini(query, results)
                st.subheader("Answer")
                st.markdown(answer)

                if results:
                    st.subheader("Relevant Code Context (Click to Expand)")
                    for i, result in enumerate(results, 1):
                        with st.expander(f"🔍 Context {i} (from {os.path.basename(result['meta']['filename'])}, score: {result['score']:.2f})"):
                            st.code(result['text'])
                            st.caption(f"File: {result['meta']['filename']}")
                else:
                    st.warning("No relevant context found")
    else:
        st.info("Please load a repository in the sidebar to get started.")

if __name__ == "__main__":
    main()