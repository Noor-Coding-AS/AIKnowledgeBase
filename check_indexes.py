# check_indexes.py
from pinecone import Pinecone
from dotenv import load_dotenv
import os

load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
indexes = pc.list_indexes().names()
print(f"Your indexes ({len(indexes)}):")
for idx in indexes:
    print(f"  → {idx}")