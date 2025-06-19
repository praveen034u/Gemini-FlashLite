from flask import Flask, request, jsonify
from langchain_core.prompts import PromptTemplate
from gemini_langchain import GeminiLLM

app = Flask(__name__)

@app.route("/generate", methods=["POST"])
def generate():
    prompt_input = request.json.get("prompt", "")
    if not prompt_input:
        return jsonify({"error": "Prompt is required"}), 400

    # Setup LangChain components
    prompt_template = PromptTemplate.from_template("You are a medical assistant. Answer the following question:\n{question}")
    llm = GeminiLLM()
    chain = prompt_template | llm

    # Generate result
    response = chain.invoke({"question": prompt_input})
    return jsonify({"response": response})
