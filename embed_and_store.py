import os
import sys
import io
import logging
from dotenv import load_dotenv
from pymongo import MongoClient
import google.generativeai as genai

# ---------- Force UTF-8 for Windows terminal ----------
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# ---------- Load environment variables ----------
load_dotenv()

# ---------- Read environment variables ----------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # Correct name for Gemini
MONGODB_URI = os.getenv("MONGODB_URI")

# ---------- Logging setup ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("embed_and_store.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# ---------- Check required environment variables ----------
if not GOOGLE_API_KEY:
    logging.critical("Missing GOOGLE_API_KEY in .env file. Cannot continue.")
    sys.exit(1)
if not MONGODB_URI:
    logging.critical("Missing MONGODB_URI in .env file. Cannot continue.")
    sys.exit(1)

class EmbedAndStore:
    """Embeds text chunks using Gemini API and stores in MongoDB."""

    def __init__(self, mongo_uri: str, db_name: str, collection_name: str, google_api_key: str):
        try:
            # MongoDB connection
            self.client = MongoClient(mongo_uri)
            self.db = self.client[db_name]
            self.collection = self.db[collection_name]

            # Gemini API configuration
            genai.configure(api_key=google_api_key)
            logging.info("MongoDB and Gemini API configured successfully.")
        except Exception as e:
            logging.error(f"Initialization failed: {e}")
            raise

    def embed_text(self, text: str):
        """Generate embeddings for given text using Gemini API."""
        try:
            model = genai.embed_content(model="models/text-embedding-004", content=text)
            return model['embedding']
        except Exception as e:
            logging.error(f"Failed to generate embedding: {e}")
            return None

    def process_chunks(self, chapter_name: str, chunks: list):
        """Process and store embeddings for text chunks."""
        for idx, chunk in enumerate(chunks, start=1):
            try:
                logging.info(f"Processing chunk {idx}/{len(chunks)}")
                embedding = self.embed_text(chunk)
                if embedding:
                    self.collection.insert_one({
                        "chapter": chapter_name,
                        "chunk_index": idx,
                        "text": chunk,
                        "embedding": embedding
                    })
                    logging.info(f"Chunk {idx} stored in MongoDB.")
                else:
                    logging.warning(f"Skipping chunk {idx} due to missing embedding.")
            except Exception as e:
                logging.error(f"Error processing chunk {idx}: {e}")

if __name__ == "__main__":
    DB_NAME = "ncert_db"
    COLLECTION_NAME = "chapter_embeddings"
    CHAPTER_NAME = "Understanding Media"

    # Load chunks from folder
    chunks_folder = "chunks"
    chunks = []
    for file in sorted(os.listdir(chunks_folder)):
        if file.endswith(".txt"):
            with open(os.path.join(chunks_folder, file), "r", encoding="utf-8") as f:
                chunks.append(f.read())

    # Process
    embedder = EmbedAndStore(MONGODB_URI, DB_NAME, COLLECTION_NAME, GOOGLE_API_KEY)
    embedder.process_chunks(CHAPTER_NAME, chunks)
    logging.info("All chunks processed.")
