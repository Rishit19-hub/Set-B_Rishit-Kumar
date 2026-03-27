# RetailMind AI Agent Solution (Set-B Case Study)

This repository contains an implementation of the RetailMind Product Intelligence AI Agent tailored specifically for the D2C fashion brand *StyleCraft*.

It acts as a conversational Product Intelligence Assistant capable of monitoring stock levels, warning about stockouts, flagging pricing issues, and parsing customer sentiment directly from raw CSV datasets.

## Architecture 

The application utilizes the **Router Agent Pattern**:
1. User types query.
2. The core AI agent (using `langchain-openai` and `gemini-1.5-flash`) predicts the precise **Intent** using Pydantic structured output classification (`INVENTORY`, `PRICING`, `REVIEWS`, `CATALOG`, `GENERAL`).
3. The specific intent dispatches the query to a dedicated domain Agent (`inventory_agent`, `pricing_agent`, etc.) equipped with bespoke native Python `@tools`.
4. Extracted results are formulated back to the user via Streamlit.

## Setup Instructions

1. **Clone the repository**: (Ensure you're inside this local dir).
2. **Setup virtual environment**:
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Configure Secrets**:
   Rename `.env.example` to `.env` and fill in your Gemini API Key.
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

## How to Run
Ensure both `retailmind_products.csv` and `retailmind_reviews.csv` are in the project root. Launch the application with:

```bash
python -m streamlit run start.py
```
