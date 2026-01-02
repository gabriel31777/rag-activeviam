# RAG Activeviam Project - AI Report Analysis

This project implements a RAG (Retrieval-Augmented Generation) system to analyze financial and sustainability reports. It uses **ChromaDB** for semantic search and **Google Gemini** to generate answers and extract specific values from complex tables.

## ⚠️ Prerequisites

1.  **Python 3.9 or higher** installed.
2.  **Google Gemini API Key**: Each team member must use their own key.
    * Get one here: [Google AI Studio](https://aistudio.google.com/app/apikey)

---

## 🚀 Installation and Setup

Follow these steps exactly to set up the environment on your machine.

### 1. Configure the API Key
For security reasons, we do not share API keys in the code.
1.  In the project root folder, create a new file named `.env` (no name before the dot).
2.  Open this file with a text editor.
3.  Paste your key in the following format and save:
    ```env
    GEMINI_API_KEY=your_api_key_here_without_quotes
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
*(You will know it worked if `(.venv)` appears at the start of the terminal line)*

### 3. Install Dependencies
With the environment activated, install the required libraries:
```bash
pip install -r requirements.txt
```

---

## ⚙️ Database Setup (Indexing)

The project already contains the raw data in `data/processed/data_ret_clean.csv`. We need to load this into the vector database (ChromaDB).

Run the indexing script **once**:

```bash
python src/03_index_chunks_meta.py
```
*This will read the CSV, chunk the text, identify the Year and Document for each row, and save everything into the `data/chroma` folder.*

---

## 🤖 How to Run and Ask Questions

The main script is `src/06_rag_generate_gemini.py`. It retrieves context from the database and uses Gemini to answer.

### Mode 1: Free Text (Explanatory Chat)
Use this mode to get an explained answer, detailing how the AI found the data. This is the **recommended mode** for complex or fragmented tables.

```bash
python src/06_rag_generate_gemini.py --q "What is the value of Scope 1 emissions for Oceana in 2021?" --answer-style free --mode chat
```

### Mode 2: Exact Value Extraction
Use this mode if you only want the raw number (useful for automation or Excel filling).

```bash
python src/06_rag_generate_gemini.py --q "What is the value of Scope 1 emissions for Oceana in 2021?" --answer-style value --mode chat
```

### Mode 3: Debug (Under the Hood)
If the answer seems wrong, use debug mode. It shows:
* Which document chunks were retrieved.
* Which metadata (Year/Doc) was identified.
* The exact prompt sent to Gemini.

```bash
python src/06_rag_generate_gemini.py --q "Your question here" --mode debug
```

---

## 🛠️ Advanced Parameters

You can tweak the script behavior with these flags:

* `--k N`: Defines how many text chunks the system reads. Default is **15**. 
    * If the answer is "I don't know", try increasing it (e.g., `--k 25`) to catch headers that are far from the data rows.
* `--max-out N`: Defines the maximum response length (tokens). Default is 8192.
* `--temp 0.1`: Model temperature. Keep it low (0.1) for precise data extraction.

## 📂 File Structure Overview

* `src/06_rag_generate_gemini.py`: **THE BRAIN.** The main RAG script with the Gemini integration.
* `src/03_index_chunks_meta.py`: Preparation script that builds the vector database.
* `data/processed/data_ret_clean.csv`: The original cleaned data file.
* `data/chroma/`: Where the local vector database is stored (do not delete unless you want to re-index).