import pandas as pd
import networkx as nx
import google.generativeai as genai
import os
import matplotlib.pyplot as plt


# Step 1: Configure Gemini API
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash-latest")

# Step 2: Load Excel Sheet
def load_excel(file_path):
    df = pd.read_excel(file_path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df = df.dropna(how='all')  # Drop rows with all NaNs
    return df

# Step 3: Build a Basic Knowledge Graph using networkx
def build_graph_from_df(df):
    G = nx.DiGraph()
    for idx, row in df.iterrows():
        node_id = f"row_{idx}"
        for col, val in row.items():
            if pd.notna(val):
                G.add_node(val, label=col)
                G.add_edge(node_id, val, relation=col)
    return G

# Step 4: Create a Textual Representation (for Gemini context)
def graph_to_text(df):
    chunks = []
    for idx, row in df.iterrows():
        row_text = f"Row {idx+1}: " + ", ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
        chunks.append(row_text)
    return "\n".join(chunks)

# Step 5: Prompt Gemini
def query_gemini(prompt_context, question):
    full_prompt = f"""You are a helpful assistant working on a table extracted from an Excel sheet.

Here is the data:
{prompt_context}

Now, answer the following question:
{question}
"""
    response = model.generate_content(full_prompt)
    return response.text

def visualize_graph(G):
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)  # positions for all nodes

    # Draw nodes with labels
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue')
    nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif")

    # Draw edges with labels
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), arrowstyle='-|>', arrowsize=20)
    edge_labels = nx.get_edge_attributes(G, 'relation')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=8)

    plt.title("Knowledge Graph from Excel Data")
    plt.axis('off')  # Hide axis
    plt.show()

# Example Usage
if __name__ == "__main__":
    excel_file = "C:/Users/baiju/Downloads/Take One Contacts.xlsx"  # 🟨 Replace with your file path
    df = load_excel(excel_file)

    print("✅ Excel file loaded and cleaned.")

    G = build_graph_from_df(df)
    print("✅ Knowledge graph created.")
    visualize_graph(G)
    graph_text = graph_to_text(df)

    while True:
        query = input("\n💬 Ask a question about the Excel sheet (or type 'exit'): ")
        if query.lower() == "exit":
            break
        answer = query_gemini(graph_text, query)
        print("\n🤖 Gemini's Answer:\n", answer)
