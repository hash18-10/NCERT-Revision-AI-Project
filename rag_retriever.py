import os
import logging
from dotenv import load_dotenv
import numpy as np
import google.generativeai as genai


logger = logging.getLogger("RAGRetrieverLogger")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    logger.warning("GOOGLE_API_KEY not set. Query embedding calls will fail until set.")
    raise SystemExit(1)

genai.configure(api_key=GOOGLE_API_KEY)

def chunk_text(text, chunk_size=300, overlap=50):
    """Split text into overlapping chunks of chunk_size."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        if chunk:
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

class FileRAGRetriever:
    """Retrieves top-k most similar chunks from a text file using Gemini embeddings."""

    def __init__(self, filepath):
        self.filepath = filepath
        self.embed_model = "models/text-embedding-004"
        self.chunks = []
        self.embeddings = []
        logger.info(f"Initializing FileRAGRetriever with file: {filepath}")
        self._load_and_embed_chunks()

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return -1.0
        return float(np.dot(a, b) / denom)

    def _embed(self, text: str):
        try:
            resp = genai.embed_content(model=self.embed_model, content=text)
            emb = resp["embedding"]
            return np.array(emb, dtype=float)
        except Exception as e:
            logger.exception("Embedding failed: %s", e)
            return None

    def _load_and_embed_chunks(self):
        if not os.path.exists(self.filepath):
            logger.error("Text file not found: %s", self.filepath)
            raise FileNotFoundError(self.filepath)
        with open(self.filepath, "r", encoding="utf-8") as f:
            text = f.read()
        self.chunks = chunk_text(text)
        self.embeddings = []
        logger.info(f"Chunking and embedding {len(self.chunks)} chunks from file.")
        for chunk in self.chunks:
            emb = self._embed(chunk)
            self.embeddings.append(emb)

    def retrieve(self, query: str, top_k: int = 3):
        logger.info(f"Retrieving top {top_k} chunks for query: {query}")
        q_emb = self._embed(query)
        if q_emb is None:
            logger.error("Could not compute query embedding.")
            return []
        scored = []
        for idx, (chunk, emb) in enumerate(zip(self.chunks, self.embeddings)):
            if emb is None:
                continue
            score = self._cosine_sim(q_emb, emb)
            scored.append({
                "chunk_id": idx,
                "filename": os.path.basename(self.filepath),
                "text": chunk,
                "score": score
            })
        scored.sort(key=lambda x: x["score"], reverse=True)
        logger.info(f"Retrieved {len(scored[:top_k])} chunks.")
        return scored[:top_k]

    def build_teaching_prompt(self, query: str, retrieved_chunks: list) -> str:
        """
        Build prompt using explicit Identity / Instructions / Example style.
        """
        chunks_text = ""
        for i, chunk in enumerate(retrieved_chunks, 1):
            chunks_text += f"{i}) {chunk['text']}\n"

        prompt = f"""
# Identity
You are a knowledgeable NCERT Class 7 Social Science teacher who explains concepts clearly and simply to students.

# Instructions
* Use only the information provided in the "Passages" section.
* Explain concepts in short, clear sentences suitable for a 12-year-old.
* Use bullet points (•) for clarity.
* Give one simple real-life example for each explanation.
* Do not add extra information not in the passages.

# Example
<user_query>
What is democracy?
</user_query>
<assistant_response>
• Democracy is a system where people choose their leaders.
• Leaders are elected through voting.
• Citizens have the right to take part in decision-making.
Example: In India, citizens vote to choose the Prime Minister.
</assistant_response>

# Passages
{chunks_text}

# Question
{query}

# Answer
"""
        return prompt.strip()


if __name__ == "__main__":
    text_file = "Understanding Media.txt"
    logger.info("Starting main program.")
    retriever = FileRAGRetriever(text_file)
    
    questions = [
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
    import random

    conversation = []  # Store tuples of (question, user_ans, model_ans, feedback, sim)

    while True:
        q = random.choice(questions)
        print(f"\nQuestion: {q}")
        user_ans = input("Your answer (type 'exit' to quit): ")
        if user_ans.strip().lower() == "exit":
            break
        results = retriever.retrieve(q, top_k=3)
        if not results:
            logger.warning("No relevant chunks found for this question.")
            print("No relevant chunks found for this question.")
            continue
        prompt = retriever.build_teaching_prompt(q, results)
        try:
            model = genai.GenerativeModel("gemini-2.0-flash-lite")
            response = model.generate_content(prompt)
            model_ans = response.text if hasattr(response, "text") else response
            print("\nModel Answer:")
            print(model_ans)
        
            user_emb = retriever._embed(user_ans)
            model_emb = retriever._embed(model_ans)
            if user_emb is not None and model_emb is not None:
                sim = retriever._cosine_sim(user_emb, model_emb)
                if sim > 0.85:
                    feedback = "Correct!"
                elif sim > 0.6:
                    feedback = "Partially correct."
                else:
                    feedback = "Incorrect or unrelated."
                print(f"\nFeedback: {feedback} (Similarity: {sim:.2f})")
            else:
                feedback = "Unable to compute similarity for feedback."
                sim = None
                print("\nFeedback: Unable to compute similarity for feedback.")
            # Store the turn in conversation history
            conversation.append({
                "question": q,
                "user_answer": user_ans,
                "model_answer": model_ans,
                "feedback": feedback,
                "similarity": sim
            })
            # Print conversation so far
            print("\n--- Conversation History ---")
            for idx, turn in enumerate(conversation, 1):
                print(f"\nTurn {idx}:")
                print(f"Q: {turn['question']}")
                print(f"Your Answer: {turn['user_answer']}")
                print(f"Model Answer: {turn['model_answer']}")
                print(f"Feedback: {turn['feedback']} (Similarity: {turn['similarity'] if turn['similarity'] is not None else 'N/A'})")
            print("\n----------------------------")
        except Exception as e:
            logger.error("Error generating answer: %s", e)
            print("Error generating answer:", e)
