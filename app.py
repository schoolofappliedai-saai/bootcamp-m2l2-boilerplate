from flask import Flask, render_template, request, session, jsonify
import requests
import os
import re
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key-change-me")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:1b")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "90"))

# Keep the dictionary tool inside app.py for teaching clarity
DICTIONARY = {
    "python": "A high-level programming language known for its simplicity and readability.",
    "flask": "A lightweight web framework for Python.",
    "api": "Application Programming Interface, a set of rules for accessing a software application.",
    "llm": "Large Language Model, an AI model trained on vast amounts of text data.",
    "agent": "An AI system that can perform tasks autonomously using tools.",
    "ollama": "A tool for running large language models locally."
}


def normalize_text(text: str) -> str:
    """
    Lowercase and remove surrounding punctuation/extra spaces.
    """
    text = text.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_lookup_key(user_message: str) -> str | None:
    """
    Detect tool-style prompts and extract the intended dictionary key.

    Supported forms:
    - lookup <key>
    - find key <key>
    - get value for <key>

    Also tries to handle inputs like:
    - lookup flask and explain it in simple words
    """
    cleaned = user_message.strip()
    lowered = cleaned.lower().strip()

    prefixes = ["lookup ", "find key ", "get value for "]
    matched_prefix = None

    for prefix in prefixes:
        if lowered.startswith(prefix):
            matched_prefix = prefix
            break

    if not matched_prefix:
        return None

    remainder = cleaned[len(matched_prefix):].strip()
    if not remainder:
        return None

    normalized_remainder = normalize_text(remainder)

    # Exact match first
    if normalized_remainder in DICTIONARY:
        return normalized_remainder

    # Try first token
    first_token = normalized_remainder.split()[0]
    if first_token in DICTIONARY:
        return first_token

    # Try finding any known key inside the remainder
    for key in DICTIONARY.keys():
        if re.search(rf"\b{re.escape(key)}\b", normalized_remainder):
            return key

    return normalized_remainder


def get_dictionary_value(key: str) -> str | None:
    """
    Return the dictionary value for the key if present.
    """
    return DICTIONARY.get(normalize_text(key))


def build_messages(user_message: str, tool_context: str | None, history: list[dict]) -> list[dict]:
    """
    Build the message list for Ollama.
    Keeps the model's general intelligence, but injects tool context when relevant.
    """
    system_prompt = (
        "You are a helpful AI assistant inside a Flask teaching demo. "
        "Keep responses concise, clear, and student-friendly. "
        "You retain your normal general intelligence for general questions. "
        "When tool context is provided, use it accurately and mention it clearly. "
        "If a dictionary key is missing, say so clearly and do not invent a value."
    )

    messages = [{"role": "system", "content": system_prompt}]

    # Only keep a short rolling context for this beginner demo
    if history:
        messages.extend(history[-8:])

    if tool_context:
        messages.append(
            {
                "role": "user",
                "content": (
                    f"User message: {user_message}\n\n"
                    f"{tool_context}\n\n"
                    "Please answer naturally. Use the tool result accurately, "
                    "but keep your normal helpful tone."
                ),
            }
        )
    else:
        messages.append({"role": "user", "content": user_message})

    return messages


def call_ollama(messages: list[dict]) -> str:
    """
    Call Ollama's local chat API.
    Raises an exception if the call fails.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
    response.raise_for_status()

    data = response.json()
    return data["message"]["content"].strip()


def generate_agent_reply(user_message: str, history: list[dict]) -> str:
    """
    Core agent logic:
    - Detect tool intent
    - If tool intent exists, look up the dictionary value
    - Pass tool context to the LLM
    - Otherwise, use the LLM normally
    """
    key = extract_lookup_key(user_message)
    tool_context = None

    if key is not None:
        value = get_dictionary_value(key)
        if value:
            tool_context = f"Dictionary tool result for key '{key}': {value}"
        else:
            tool_context = f"Dictionary tool result: no value found for key '{key}'."

    messages = build_messages(user_message, tool_context, history)
    return call_ollama(messages)


@app.route("/")
def index():
    history = session.get("history", [])
    return render_template(
        "index.html",
        history=history,
        model_name=OLLAMA_MODEL,
    )


@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.form.get("message", "").strip()

    if not user_message:
        return jsonify({"error": "No message provided."}), 400

    history = session.get("history", [])

    # Store user message first so it appears immediately in the chat flow
    history.append({"role": "user", "content": user_message})

    try:
        # Pass prior history only, not the just-added user message again
        reply = generate_agent_reply(user_message, history[:-1])
    except requests.exceptions.RequestException as exc:
        reply = (
            "I could not reach the local Ollama model. "
            "Please make sure Ollama is running and the selected model is available.\n\n"
            f"Technical detail: {exc}"
        )
    except KeyError:
        reply = "The response from Ollama did not match the expected format."
    except Exception as exc:
        reply = f"Something went wrong while generating the response: {exc}"

    history.append({"role": "assistant", "content": reply})

    # Keep session history small for a lightweight demo
    session["history"] = history[-12:]
    session.modified = True

    return jsonify({"response": reply})


@app.route("/clear", methods=["GET", "POST"])
def clear():
    session["history"] = []
    session.modified = True
    return jsonify({"status": "cleared"})


if __name__ == "__main__":
    app.run(debug=True)