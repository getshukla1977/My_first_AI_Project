import os
import streamlit as st
from datetime import datetime
from groq import Client
from dotenv import dotenv_values, load_dotenv

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DOTENV_PATH = os.path.join(ROOT_DIR, ".env")
load_dotenv(dotenv_path=DOTENV_PATH, override=True)

env_overrides = dotenv_values(dotenv_path=DOTENV_PATH) if os.path.exists(DOTENV_PATH) else {}
API_KEY = (
    os.getenv("GROQ_API_KEY")
    or os.getenv("OPENROUTER_API_KEY")
    or env_overrides.get("GROQ_API_KEY")
    or env_overrides.get("OPENROUTER_API_KEY")
)
MODEL_OPTIONS = [
    "openai/gpt-oss-20b",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "llama-3.3-70b-versatile",
    "gemma2-9b-it",
]
MODEL = MODEL_OPTIONS[0]
client = None
if API_KEY:
    client = Client(
        base_url="https://api.groq.com",
        api_key=API_KEY,
    )

def analyze_product(product_name: str, extra_context: str = "") -> str:
    current_date = datetime.now().strftime("%Y-%m-%d")
    system_prompt = (
        "You are a senior product and business analyst. You write clear, practical, "
        "well-structured product analysis reports for founders and business teams."
    )
    user_prompt = f"""
Write a detailed product analysis report for: {product_name}
Current date: {current_date}

Please cover the following in one flowing, well-organized report (use markdown headings and bullet points where helpful):
- Market demand and ideal customer profile.
- Marketing strategies to reach the widest possible audience (at least 5 points).
- Technology and manufacturing feasibility / key requirements (at least 5 points).
- Business model: scalability and revenue streams (at least 5 points).
- A concise business plan, goals, and launch timeline.

Keep it insightful and actionable.

{extra_context}
"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    return response.choices[0].message.content.strip() if response.choices and response.choices[0].message else ""

def main():
    st.set_page_config(page_title="Product Analysis", page_icon="🧠", layout="wide")
    st.title("Product Analysis Generator")
    st.write(
        "Generate a practical product analysis report for your idea using Groq and Streamlit."
    )

    with st.sidebar:
        st.header("Product Analysis Input")
        product_name = st.text_input("Product, service, or idea name")
        extra_context = st.text_area(
            "Additional context (optional)",
            help="Add details such as target market, pricing, or product category.",
        )
        st.markdown(
            "---\n"
            "### How to use\n"
            "1. Enter the product name.\n"
            "2. Add any optional context.\n"
            "3. Click Generate.\n"
            "4. The report will appear below.\n"
        )
        if not API_KEY:
            st.warning(
                "GROQ_API_KEY is not set. Create a `.env` file with `GROQ_API_KEY=your_key`."
            )

    if st.button("Generate Product Analysis"):
        if not product_name.strip():
            st.error("Please enter a product, service, or idea name first.")
            return
        if not client:
            st.error("Missing GROQ_API_KEY. Set it in the .env file or environment.")
            return

        with st.spinner("Creating product analysis report..."):
            try:
                report = analyze_product(product_name, extra_context)
                st.success("Analysis complete.")
                st.markdown(report)
            except Exception as exc:
                st.error("Failed to generate the report.")
                st.write(exc)

if __name__ == "__main__":
    main()

