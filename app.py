from flask import Flask, render_template, request, session, jsonify
from openai import OpenAI
import os
import re
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key-change-me")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
OPENAI_TIMEOUT = int(os.getenv("OPENAI_TIMEOUT", "90"))

client = OpenAI(api_key=OPENAI_API_KEY)

# Kept the dictionary tool inside app.py for teaching clarity
DICTIONARY = {
    "python": "A high-level programming language known for its simplicity and readability.",
    "flask": "A lightweight web framework for Python.",
    "api": "Application Programming Interface, a set of rules for accessing a software application.",
    "llm": "Large Language Model, an AI model trained on vast amounts of text data.",
    "agent": "An AI system that can perform tasks autonomously using tools.",
    "ollama": "A tool for running large language models locally.",
    "openai": "A platform and API for using large language models in applications."
}

# Records tool: dictionary of dictionaries
RECORDS = {
    "aaditya": {"name": "Aaditya Jain", "age": 25, "city": "New York"},
    "aabhas": {"name": "Aabhas Jaiswal", "age": 22, "city": "Los Angeles"},
    "anmol": {"name": "Anmol Shakya", "age": 24, "city": "Chicago"},
    "lucky": {"name": "Lucky Shivhare", "age": 26, "city": "Houston"},
    "aadya": {"name": "Aadya Singh", "age": 23, "city": "Phoenix"},
    "jairaj": {"name": "Jairaj Singh Rathore", "age": 27, "city": "Philadelphia"},
    "kavyansh": {"name": "Kavyansh Singh Rajput", "age": 24, "city": "San Francisco"},
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


def find_record_person(user_message: str) -> str | None:
    """
    Try to find which person from RECORDS is being mentioned.
    Returns the canonical RECORDS key if found.
    """
    normalized_message = normalize_text(user_message)

    for person_key, details in RECORDS.items():
        # Match the short key like "aaditya"
        if re.search(rf"\b{re.escape(normalize_text(person_key))}\b", normalized_message):
            return person_key

        # Match the full name like "Aaditya Jain"
        full_name = details.get("name")
        if full_name and normalize_text(full_name) in normalized_message:
            return person_key

    return None


def extract_record_field(user_message: str) -> str | None:
    """
    Detect which field about the person is being asked.
    """
    normalized_message = normalize_text(user_message)

    field_aliases = {
        "name": ["name", "full name"],
        "age": ["age", "old"],
        "city": ["city", "live", "lives", "from", "location"],
    }

    for field, aliases in field_aliases.items():
        for alias in aliases:
            if re.search(rf"\b{re.escape(alias)}\b", normalized_message):
                return field

    return None


def get_record_detail(person_key: str, field: str):
    """
    Return a particular field for a person from RECORDS.
    """
    person_data = RECORDS.get(person_key)
    if not person_data:
        return None

    return person_data.get(field)


def build_messages(user_message: str, tool_context: str | None, history: list[dict]) -> list[dict]:
    """
    Build the message list for OpenAI.
    Keeps the model's general intelligence, but injects tool context when relevant.
    """
    system_prompt = (
        "You are a helpful AI assistant inside a Flask teaching demo. "
        "Keep responses concise, clear, and student-friendly. "
        "You retain your normal general intelligence for general questions. "
        "When tool context is provided, use it accurately and mention it clearly. "
        "If a dictionary key is missing, say so clearly and do not invent a value. "
        "If a person record is found, use only the provided record data and do not invent missing details."
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


def call_openai(messages: list[dict]) -> str:
    """
    Call OpenAI Responses API.
    Raises an exception if the call fails.
    """
    response = client.responses.create(
        model=OPENAI_MODEL,
        input=messages,
        timeout=OPENAI_TIMEOUT,
    )

    return response.output_text.strip()


def generate_agent_reply(user_message: str, history: list[dict]) -> str:
    """
    Core agent logic:
    1. Detect dictionary lookup intent
    2. Detect person/record lookup intent
    3. Use OpenAI normally for all other prompts
    """

    # --- Dictionary lookup ---
    key = extract_lookup_key(user_message)
    if key is not None:
        value = get_dictionary_value(key)
        if value:
            return f"The dictionary value for '{key}' is: {value}"
        return f"No dictionary value found for key '{key}'."

    # --- Records lookup ---
    person_key = find_record_person(user_message)
    field = extract_record_field(user_message)

    if person_key and field:
        value = get_record_detail(person_key, field)
        full_name = RECORDS[person_key].get("name", person_key)

        if value is not None:
            return f"{full_name}'s {field} is {value}."
        return f"I found {full_name}, but I could not find the field '{field}'."

    if person_key and not field:
        person_data = RECORDS.get(person_key, {})
        available_fields = ", ".join(person_data.keys())
        full_name = person_data.get("name", person_key)
        return f"I found {full_name}. Available details are: {available_fields}."

    # --- General AI fallback ---
    messages = build_messages(user_message, None, history)
    return call_openai(messages)


@app.route("/")
def index():
    history = session.get("history", [])
    return render_template(
        "index.html",
        history=history,
        model_name=OPENAI_MODEL,
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
    except Exception as exc:
        reply = (
            "I could not reach the OpenAI API. "
            "Please make sure your OPENAI_API_KEY is set correctly and your internet connection is working.\n\n"
            f"Technical detail: {exc}"
        )

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
