# Minimal AI Agent with Flask

This project demonstrates the difference between a normal LLM chatbot and a minimal AI agent that can use tools.

## What is an AI Agent?

An AI agent is an AI system that can perform tasks autonomously, often by using tools to gather information or perform actions. In this example, the agent can look up definitions from a built-in dictionary when prompted with specific commands.

## Setup

1. (Optional) Create a virtual environment:

   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Copy environment file:

   ```bash
   copy .env.example .env
   ```

   Edit `.env` to set your `FLASK_SECRET_KEY` to a random string.

4. Start Ollama:

   Make sure Ollama is running locally.

5. Pull the model:

   ```bash
   ollama pull gemma3:1b
   ```

6. Run the app:

   ```bash
   python app.py
   ```

   Open http://localhost:5000 in your browser.

## Test Prompts

- `What is Flask?` (normal LLM response)
- `lookup python`
- `lookup flask`
- `find key llm`
- `get value for api`
- `lookup django` (should say not found)
- `lookup flask and explain it in simple words`