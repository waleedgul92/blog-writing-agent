# blog-writing-agent

A highly concurrent, multi-agent blog generation pipeline built with **LangGraph**, **FastAPI**, and **Streamlit**. Powered by Google's **Gemini 2.5 Flash** model.

This application takes a single topic and intelligently decides whether to search the live web for recent information or rely on its internal knowledge. It then plans an outline and spawns multiple AI workers in parallel to write individual sections, drastically reducing generation time.

---

## ✨ Features

- 🧠 **Intelligent Routing** — Automatically detects if a topic requires live internet research (e.g., news, volatile pricing) or if it is an evergreen concept.
- 🔍 **Web Research Integration** — Uses Tavily Search to gather facts, deduplicate sources, and strictly ground the AI's writing in real evidence with citations.
- ⚡ **Parallel Generation** — Uses LangGraph's `Send` API (Fanout) to write up to 4 blog sections simultaneously.
- 🎨 **Clean UI** — A Streamlit frontend that provides real-time progress, markdown previews, and easy downloads for your generated content.
- 📂 **History Management** — Automatically saves past blogs to a local directory and lets you reload them into the UI with a single click.

---

## 🏗️ Architecture

### System Diagram



<img width="1408" height="768" alt="image" src="https://github.com/user-attachments/assets/136b15c4-fa92-47dc-a3d8-13e247d6ca86" />

### How It Works

The pipeline follows a **route → research → plan → fanout → reduce** pattern:

1. **Router** — Classifies the topic as `web_search` or `general`.
2. **Researcher** *(conditional)* — Fetches and deduplicates live sources via Tavily.
3. **Orchestrator** — Produces a structured outline with up to 4 sections.
4. **Workers** — Each section is written in parallel using LangGraph's `Send` API.
5. **Reducer** — Assembles all sections into a single cohesive blog post.

### Project Structure
```
blog/
├── app/
│   ├── schemas.py      # Pydantic models & LangGraph State definition
│   ├── nodes.py        # Core AI logic — Router, Researcher, Orchestrator, Worker, Reducer
│   ├── backend.py      # FastAPI app wrapping LangGraph execution
│   └── main.py         # Uvicorn entry point, file reading & zipping endpoints
├── frontend.py         # Streamlit UI — communicates with backend via REST
├── .env                # API keys (not committed to version control)
└── requirements.txt
```

---

## ⚙️ Prerequisites & Setup

### 1. Environment Variables

Create a `.env` file in the root directory of the project and add your API keys:
```env
# Required: Google Gemini API Key
GEMINI_API_KEY="AIzaSyYourActualKeyHere..."

# Required for Web Research: Tavily Search API Key
TAVILY_API_KEY="tvly-YourActualKeyHere..."
```

> ⚠️ Never commit your `.env` file to version control. Add it to `.gitignore`.

### 2. Install Dependencies

Ensure you have **Python 3.9+** installed. Then install all required packages:
```bash
pip install langgraph langchain-google-genai langchain-community tavily-python fastapi uvicorn streamlit pydantic python-dotenv pandas requests
```

---

## 🚀 How to Run

Because the backend and frontend are decoupled, you need to run them in **two separate terminal windows**.

### Terminal 1 — Start the Backend (FastAPI)

Run the Uvicorn server to host the LangGraph API on port `8000`:
```bash
uvicorn main:api --reload --port 8000
```

### Terminal 2 — Start the Frontend (Streamlit)

Run the Streamlit application to launch the user interface:
```bash
streamlit run frontend.py
```

Then open your browser and navigate to **http://localhost:8501**

> 💡 If your terminal throws a `gio: Operation not supported` error, simply open the URL manually in your browser.

---

## 📄 License

MIT
