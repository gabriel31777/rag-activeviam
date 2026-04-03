# RAG Activeviam Project - AI Financial Report Analysis

This project implements a high-fidelity RAG (Retrieval-Augmented Generation) system to extract data from raw PDF financial and sustainability reports. It uses **ChromaDB** for storage, **PyMuPDF** for table-aware PDF parsing, and an agentic loop (Groq/Gemini) to navigate complex financial data.

## ⚠️ Prerequisites

1.  **Python 3.10 or higher** installed.
2.  **API Keys**: You need keys from Groq and/or Google Gemini.
    *   **Groq (Primary)**: [Groq Console](https://console.groq.com/keys)
    *   **Gemini (Fallback)**: [Google AI Studio](https://aistudio.google.com/app/apikey)

---

## 🚀 Installation and Setup

### 1. Configure the API Keys
For security reasons, do **not** commit your API keys. 
1.  Copy `.env.example` to a new file named `.env`:
    ```bash
    cp .env.example .env
    ```
2.  Open `.env` and fill in your keys:
    ```env
    GROQ_API_KEY=your_groq_key_here
    GEMINI_API_KEY=your_gemini_key_here
    ```

### 2. Create Virtual Environment
Open your terminal in the project folder and run:

**Windows:**
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## ⚙️ PDF Indexing (The Database)

We index raw PDF documents from `data/raw/Structured data/`. To rebuild the database from scratch:

```bash
python src/12_index_pdfs_full.py
```
*This uses PyMuPDF to extract text while preserving table structures and maps them into the `activeviam_pdfs_v1` collection.*

---

## 🤖 How to Use

### 1. Web Interface (The Dashboard)
The easiest way to use the system is via the web app:
```bash
python app.py
```
Then open `http://127.0.0.1:5000` in your browser.

### 2. Accuracy Evaluation (Benchmark)
To test the agent's performance against the gold dataset (excluding zero-value responses):
```bash
python src/08_eval_agent.py --limit 10
```
*Note: The script includes a 30-second delay between questions to respect API rate limits.*

---

## 🛠️ Key Scripts

*   `src/09_rag_agent_groq.py`: The core agent logic. It manages the thinking loop, tool calls, and API fallbacks.
*   `src/12_index_pdfs_full.py`: The PDF indexer. Use this when you add new PDF files to the data folder.
*   `src/08_eval_agent.py`: Automated evaluation script to measure accuracy.
*   `app.py`: Flask server for the UI.

## 📂 File Structure

*   `data/raw/Structured data/`: Put your PDF reports here.
*   `.env`: Your private API keys (ignored by Git).
*   `.env.example`: Template for environment variables.
*   `chroma/`: Local vector database storage.