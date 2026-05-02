# Medical Chatbot

This project now keeps only the main Streamlit application and the files it needs to run.

## Main App

Run:

```bash
streamlit run medical_chatbot_ui.py
```

## Setup

1. Create or activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Add your Groq API key in `key.env`:

```env
GROQ_API_KEY=your_groq_api_key
```

4. Start the app:

```bash
streamlit run medical_chatbot_ui.py
```

5. Open the browser at:

```text
http://127.0.0.1:8501
```

## Required Runtime Files

- `medical_chatbot_ui.py`
- `custom_favicon.py`
- `assets/`
- `vectorstore/`
- `requirements.txt`
- `key.env`

## Notes

- The vector database inside `vectorstore/db_faiss/` must remain present.
- If `key.env` is missing or the API key is invalid, the app will not answer queries.
