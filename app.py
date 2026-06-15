import streamlit as st
import os
from dotenv import load_dotenv

# Local development: load API key from .env
load_dotenv()

# Try Streamlit Cloud secrets first; fall back to env vars
try:
    api_key = st.secrets["ANTHROPIC_API_KEY"]
except (KeyError, FileNotFoundError):
    api_key = os.getenv("ANTHROPIC_API_KEY")

# Page config
st.set_page_config(
    page_title="PM Agent #2 — Research Synthesizer",
    page_icon="📋",
    layout="centered"
)

# Header
st.title("📋 Research Synthesizer")
st.markdown("*Drop 3-5 user interview transcripts. Get a structured synthesis report.*")

# Setup verification
st.markdown("---")
st.subheader("Setup verification")

if api_key:
    st.success(f"✅ Anthropic API key loaded successfully (key starts with `{api_key[:12]}...`)")
else:
    st.error("❌ Anthropic API key not found. Check your .env file (local) or Streamlit Cloud secrets (deployed).")

st.info("This is the infrastructure-only placeholder. Real agent features ship tomorrow.")

# Footer
st.markdown("---")
st.markdown("*PM Agent #2 of 24 in [StreamMind](https://github.com/shreya-patel-PM/streammind-pm2-research-synthesizer)*")