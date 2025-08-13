import streamlit as st
from rag_retriever import FileRAGRetriever
import google.generativeai as genai
import os
import logging
import numpy as np
import random

st.set_page_config(page_title="NCERT RAG Q&A", layout="wide")

response_logger = logging.getLogger("ResponseLogger")
response_logger.setLevel(logging.INFO)
resp_handler = logging.FileHandler("responses.log", encoding="utf-8")
resp_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
resp_handler.setFormatter(resp_formatter)
if not response_logger.hasHandlers():
    response_logger.addHandler(resp_handler)

@st.cache_resource
def load_retriever():
    text_file = "Understanding Media.txt"
    return FileRAGRetriever(text_file)

retriever = load_retriever()

understanding_media_questions = [
    "What is media?",
    "Why is television called mass media?",
    "How has technology changed the media?",
    "What is the difference between print media and electronic media?",
    "Why does mass media need a lot of money?",
    "How does media earn money?",
    "What is the role of media in a democracy?",
    "What is a balanced media report?",
    "Why is independent media important in a democracy?",
    "What is censorship?",
    "How does media set the agenda?",
    "Give an example of how media influenced public awareness.",
    "Why do advertisements appear so often on TV?",
    "What is social advertising?",
    "How can media reports be one-sided?",
    "What questions should we ask to analyze a news report?",
    "How does television influence our view of the world?",
    "Why is it important to know both sides of a story?",
    "What is the relationship between media and business?",
    "Why should we be active viewers of media?"
]

st.title("NCERT Class 7 Social Science AI Revision")
st.write("You will be asked a question from the 'Understanding Media' chapter. Type your answer and get instant feedback!")

if "current_q" not in st.session_state:
    st.session_state.current_q = random.choice(understanding_media_questions)

if st.button("Next Question"):
    st.session_state.current_q = random.choice(understanding_media_questions)

question = st.session_state.current_q
st.markdown(f"### Question: {question}")

user_answer = st.text_area("Your Answer:", key="user_answer")

if "conversation" not in st.session_state:
    st.session_state.conversation = []

if st.button("Submit Answer") and user_answer.strip():
    results = retriever.retrieve(question, top_k=3)
    if not results:
        st.warning("No relevant chunks found for this question.")
        response_logger.info(f"Question: {question}\nUser Answer: {user_answer}\nModel Answer: No relevant chunks found.")
    else:
        prompt = retriever.build_teaching_prompt(question, results)
        try:
            model = genai.GenerativeModel("gemini-2.0-flash-lite")
            response = model.generate_content(prompt)
            model_answer = response.text if hasattr(response, "text") else str(response)
            st.markdown("**Model Answer:**")
            st.write(model_answer)
            
            user_emb = retriever._embed(user_answer)
            model_emb = retriever._embed(model_answer)
            if user_emb is not None and model_emb is not None:
                sim = retriever._cosine_sim(user_emb, model_emb)
                if sim > 0.85:
                    feedback = "✅ Correct!"
                elif sim > 0.6:
                    feedback = "🟡 Partially correct."
                else:
                    feedback = "❌ Incorrect or unrelated."
                st.markdown(f"**Feedback:** {feedback} (Similarity: {sim:.2f})")
            else:
                st.info("Could not compute feedback for your answer.")
                sim = None

            
            st.session_state.conversation.append({
                "question": question,
                "user_answer": user_answer,
                "model_answer": model_answer,
                "feedback": feedback if user_emb is not None and model_emb is not None else "Could not compute feedback",
                "similarity": sim if user_emb is not None and model_emb is not None else "N/A"
            })

            response_logger.info(f"Question: {question}\nUser Answer: {user_answer}\nModel Answer: {model_answer}\nSimilarity: {sim if 'sim' in locals() else 'N/A'}")
        except Exception as e:
            st.error(f"Error generating answer: {e}")
            response_logger.error(f"Question: {question}\nError: {e}")

if st.session_state.conversation:
    st.markdown("---")
    st.markdown("### Conversation History")
    for idx, turn in enumerate(st.session_state.conversation, 1):
        st.markdown(f"**Turn {idx}:**")
        st.markdown(f"- **Question:** {turn['question']}")
        st.markdown(f"- **Your Answer:** {turn['user_answer']}")
        st.markdown(f"- **Model Answer:** {turn['model_answer']}")
        st.markdown(f"- **Feedback:** {turn['feedback']} (Similarity: {turn['similarity']})")
        st.markdown("---")
