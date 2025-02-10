# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from rag_workflow import run_agentic_rag_conversation

app = Flask(__name__)
CORS(app) 

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    # data should be { messages: [ {role, content} ] }.....
    conversation = data.get("messages", [])

    # Pass the entire conversation to the RAG workflow
    try:
        final_response = run_agentic_rag_conversation(conversation)
        return jsonify({"message": final_response})
    except Exception as e:
        print("Error running the agentic RAG conversation:", str(e))
        return jsonify({"message": "An error occurred on the server."}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
