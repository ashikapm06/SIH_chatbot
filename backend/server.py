# server.py
import os
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS

from mchat import generate_mangalore_transport_data, generate_govt_transport_data, CombinedRAG

print("Starting server...")

app = Flask(__name__)
CORS(app)

# Global variables
rag = None
user_df = None
govt_df = None

def init_rag():
    """Initialize datasets and RAG lazily."""
    global rag, user_df, govt_df
    if rag is not None:
        return rag

    try:
        print("Generating user transport data (may take a few seconds)...")
        user_df = generate_mangalore_transport_data(1200)  # adjust row count if needed

        locs_info = {}
        for loc in sorted(set(user_df['Origin'].tolist() + user_df['Destination'].tolist())):
            subset = user_df[(user_df['Origin'] == loc) | (user_df['Destination'] == loc)]
            if not subset.empty:
                mapping = {'Low': 0.2, 'Medium': 0.55, 'High': 0.9}
                avg_rush = subset['RushLevel'].map(mapping).mean()
                peak_hours = subset['Hour'].value_counts().head(3).index.tolist()
            else:
                avg_rush = 0.5
                peak_hours = [8, 17]
            locs_info[loc] = {'rush_prob': float(avg_rush), 'peak_hours': peak_hours, 'type': 'mixed'}

        print("Generating government dataset...")
        govt_df = generate_govt_transport_data(locs_info)

        print("Initializing RAG...")
        GEMINI_KEY = os.getenv("GEMINI_API_KEY")
        rag = CombinedRAG(GEMINI_KEY, user_df, govt_df)
        print("RAG ready.")
    except Exception as e:
        print("Failed to initialize RAG:", str(e))
        traceback.print_exc()
        rag = None

    return rag

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True})

@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        init_rag()
        if rag is None:
            return jsonify({"error": "RAG not initialized"}), 500

        data = request.get_json(force=True)
        role = data.get("role", "user")
        message = data.get("message") or data.get("query") or ""
        if not message:
            return jsonify({"error": "message required"}), 400

        answer = rag.answer(message)
        return jsonify({"answer": answer, "followupOffers": []})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Server starting on port {port}...")
    app.run(host="0.0.0.0", port=port)
