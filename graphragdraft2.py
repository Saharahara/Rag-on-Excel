# graphrag_excel.py
import os
import pandas as pd
import networkx as nx
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import google.generativeai as genai

# ----------------------- SETUP -----------------------

EXCEL_PATH = "C:/Users/baiju/Downloads/Take One Contacts.xlsx"  # <-- Replace with your file
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
genai.configure(api_key=api_key)
gemini = genai.GenerativeModel("gemini-1.5-flash-latest")
# Load Gemini model

# Embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ----------------- STEP 1: ROBUST PARSING -----------------

def parse_excel_with_fallback(path):
    for header in range(3):
        try:
            df = pd.read_excel(path, header=header)
            if df.columns.is_unique:
                return df
        except Exception:
            continue
    raise Exception("Failed to parse Excel with a usable header.")

df = parse_excel_with_fallback(EXCEL_PATH)
df.fillna("N/A", inplace=True)

# ----------------- STEP 2: EXTRACT ENTITY-RELATION TRIPLETS -----------------

def extract_triplets_with_gemini(row_dict):
    row_text = ", ".join(f"{k}: {v}" for k, v in row_dict.items())
    prompt = f"""
Given this row from an Excel sheet:
{row_text}

Extract semantic relationships as triplets in the format:
(Entity1, Relationship, Entity2)

Return only a list of such triplets.
"""
    try:
        response = gemini.generate_content(prompt)
        return response.text.strip().split("\n")
    except Exception:
        return []

triplets = set()
for _, row in df.iterrows():
    try:
        row_dict = row.to_dict()
        triplet_lines = extract_triplets_with_gemini(row_dict)
        for t in triplet_lines:
            t = t.strip().strip("()")
            parts = [x.strip() for x in t.split(",")]
            if len(parts) == 3:
                triplets.add(tuple(parts))
    except Exception:
        continue

# ----------------- STEP 3: BUILD GRAPH -----------------

G = nx.MultiDiGraph()
for head, rel, tail in triplets:
    G.add_edge(head, tail, label=rel)

# ----------------- STEP 4: VECTOR INDEXING -----------------

# Chunk each row into semantic text blobs
chunks = []
chunk_texts = []

for _, row in df.iterrows():
    row_text = ", ".join(f"{col}: {row[col]}" for col in df.columns)
    chunks.append(row_text)
    chunk_texts.append(row_text)

# Embed the chunks
embeddings = embedder.encode(chunk_texts, convert_to_numpy=True)
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# ----------------- STEP 5: HYBRID QUERY PIPELINE -----------------

def search_semantic_context(query, top_k=5):
    query_embedding = embedder.encode([query])
    D, I = index.search(query_embedding, top_k)
    return [chunks[i] for i in I[0]]

def extract_graph_context(entity, k=2):
    try:
        neighbors = nx.single_source_shortest_path_length(G, entity, cutoff=k)
        lines = []
        for n in neighbors:
            for nbr in G[n]:
                for edge_data in G[n][nbr].values():
                    lines.append(f"{n} --{edge_data['label']}--> {nbr}")
        return lines
    except Exception:
        return []

def ask_question(query):
    # Try to extract main entity from the query using Gemini
    entity_prompt = f"Extract the main entity from this question: {query}"
    entity = gemini.generate_content(entity_prompt).text.strip().split("\n")[0]

    graph_context = extract_graph_context(entity)
    semantic_context = search_semantic_context(query)

    hybrid_context = "\n".join(graph_context + semantic_context)

    final_prompt = f"""
You are a data assistant. Based on the following information from an Excel-derived knowledge base,
answer the user's question as accurately as possible.

Context:
{hybrid_context}

Question:
{query}
"""
    for _ in range(3):
        try:
            response = gemini.generate_content(final_prompt)
            return response.text.strip()
        except ResourceExhausted:
            print("⏳ Gemini quota hit while generating answer. Waiting 60s...")
            time.sleep(60)

    return "⚠️ Unable to generate a response due to quota limits."

# ----------------- USAGE EXAMPLE -----------------

if __name__ == "__main__":
    print("📄 Excel loaded, graph and vectors built.")
    while True:
        user_query = input("\n🔍 Ask a question (or type 'exit'): ")
        if user_query.lower() == "exit":
            break
        answer = ask_question(user_query)
        print(f"\n💡 Answer: {answer}")
