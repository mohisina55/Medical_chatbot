# 🩺 Medical Chatbot with Gemini + TF-IDF

This is an intelligent **Medical Chatbot** built using **Streamlit**, **Gemini 1.5 (Google Generative AI)**, and **TF-IDF + Cosine Similarity** for retrieving answers from a medical knowledge base. The bot can answer user queries, refine responses with Gemini, and maintain a chat history.

---

## 🚀 Features

- 🧠 **Hybrid NLP system**: Combines keyword-matching (TF-IDF + cosine similarity) and generative AI (Gemini).
- 📚 **Knowledge base support**: Retrieves the closest predefined answer if relevant.
- 🤖 **Gemini 1.5 refinement**: Enhances matched responses or generates new ones if no match is found.
- 💬 **Conversation history**: Displayed interactively in a clean, readable chat format.
- 🧾 **Fallback logic**: Automatically switches to AI-based response when no close match is found.

---

## 📁 File Structure

```bash
medical-chatbot/
│
├── med_bot_data.csv              # Medical knowledge base (with 'short_question' and 'short_answer' columns)
├── medical_chatbot.py            # Main Streamlit application
├── README.md                     # Project documentation (this file)

```
## 📦 Requirements
Install the following Python packages:

```bash
pip install streamlit pandas scikit-learn google-generativeai
```

