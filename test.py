import os
from pymongo import MongoClient
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

# Get Mongo URI
mongo_uri = os.getenv("MONGODB_URI")
if not mongo_uri:
    raise ValueError("MONGODB_URI not found in .env")

# Connect to MongoDB
client = MongoClient(mongo_uri)

# Show databases & collections
print("Databases:", client.list_database_names())
db = client["rag"]  # Change if DB name is different
print("Collections in 'rag':", db.list_collection_names())

# Access 'chunks' collection
coll = db["chunks"]
print("Document count in 'chunks':", coll.count_documents({}))

# Get one document without projection
sample_doc = coll.find_one()
print("\n=== Sample Document ===")
print(sample_doc)

# Print all keys present
if sample_doc:
    print("\nKeys in sample document:", list(sample_doc.keys()))
else:
    print("\nNo documents found in 'chunks'")
