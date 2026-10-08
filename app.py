import os

import boto3
import streamlit as st
from dotenv import load_dotenv
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import FAISS

from bedrock import DEFAULT_MODEL_ID, EMBEDDING_MODEL_ID, BedrockLLM, GuardrailBlocked
from pipeline import Turn, UnsupportedImage, handle_turn

st.set_page_config(page_title="GourmetGuide AI", page_icon="🍽️", layout="wide", initial_sidebar_state="expanded")

load_dotenv()


def setting(name, default=None):
    """Environment first, then Streamlit secrets (Streamlit Cloud), then the default."""
    if os.environ.get(name):
        return os.environ[name]
    try:
        return st.secrets.get(name, default)
    except Exception:  # no secrets.toml locally
        return default


# boto3 reads credentials from the environment, so bridge any keys kept in Streamlit secrets.
for key in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_DEFAULT_REGION"):
    value = setting(key)
    if value:
        os.environ[key] = value

REGION = setting("AWS_DEFAULT_REGION", "us-east-1")
INDEX_DIR = "output/faiss_index"


@st.cache_resource
def load_resources():
    client = boto3.client("bedrock-runtime", region_name=REGION)
    llm_settings = {
        "model_id": setting("BEDROCK_MODEL_ID", DEFAULT_MODEL_ID),
        "guardrail_id": setting("BEDROCK_GUARDRAIL_ID"),
        "guardrail_version": setting("BEDROCK_GUARDRAIL_VERSION", "DRAFT"),
    }
    embeddings = BedrockEmbeddings(client=client, model_id=EMBEDDING_MODEL_ID)
    # The index is built by this repo's ingest.py, never downloaded, so loading its pickle is safe.
    db = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    cuisines = sorted({doc.metadata["cuisine"] for doc in db.docstore._dict.values()})
    return client, llm_settings, db, cuisines


if not os.path.exists(INDEX_DIR):
    st.error("FAISS index not found. Run `python ingest.py` first.")
    st.stop()

try:
    client, llm_settings, db, cuisines = load_resources()
except Exception as e:
    st.error(f"Error initializing AWS clients or the vector index: {e}")
    st.stop()

# --- SESSION STATE ---
st.session_state.setdefault("turns", [])  # [(user_text, Turn)]
st.session_state.setdefault("uploader_key", 0)

# --- SIDEBAR ---
with st.sidebar:
    st.title("🍽️ Settings")
    st.markdown("---")
    st.subheader("📷 Visual Search")
    uploaded_image = st.file_uploader(
        "Upload a food image to find similar dishes:",
        type=["png", "jpg", "jpeg", "gif", "webp"],
        key=f"uploader_{st.session_state['uploader_key']}",
    )
    if uploaded_image:
        st.image(uploaded_image, caption="Preview", use_container_width=True)
    st.markdown("---")
    if st.button("🗑️ Clear Chat History", type="primary"):
        st.session_state["turns"] = []
        st.rerun()

# --- CHAT ---
st.title("GourmetGuide AI")
st.markdown("### *Your Personal Culinary Concierge*")
st.markdown("Ask me about cuisines, budgets, diets or allergies, or upload a photo of a dish you love!")
st.divider()

for user_text, turn in st.session_state["turns"]:
    with st.chat_message("user"):
        st.markdown(user_text)
    with st.chat_message("assistant"):
        st.markdown(turn.text)
        for metadata, summary in turn.recommendations:
            col1, col2 = st.columns([1, 3])
            with col1:
                st.image("data/" + metadata["image_path"], use_container_width=True)
            with col2:
                st.subheader(metadata.get("menu_item_name", "Unknown Dish"))
                st.markdown(summary)
                st.caption(f"📍 {metadata.get('restaurant_name', 'N/A')} | ⭐ {metadata.get('average_rating', 'N/A')}")
                m1, m2, m3 = st.columns(3)
                m1.metric("Price", f"${metadata.get('price', '?')}")
                m2.metric("Calories", metadata.get("calories", "N/A"))
                m3.metric("Serves", str(metadata.get("serves", "1")).strip('"'))
                warnings = str(metadata.get("dietary_warnings", "")).strip()
                if warnings:
                    st.caption(f"⚠️ {warnings}")
            st.divider()

with st.form(key="chat_form", clear_on_submit=True):
    col_input, col_btn = st.columns([6, 1])
    with col_input:
        user_input = st.text_input(
            "Type your request...", placeholder="E.g., something spicy and vegetarian under $10, no nuts", key="input_field"
        )
    with col_btn:
        st.write("")
        st.write("")
        submit_button = st.form_submit_button("Send 🚀")

if submit_button and (user_input or uploaded_image):
    image_bytes = uploaded_image.getvalue() if uploaded_image else None
    history = [(u, t.text) for u, t in st.session_state["turns"]]
    llm = BedrockLLM(client, **llm_settings)
    try:
        with st.spinner("Finding dishes..."):
            turn = handle_turn(llm, db, cuisines, user_input, image_bytes, history)
    except (UnsupportedImage, GuardrailBlocked) as e:
        turn = Turn(text=getattr(e, "message", str(e)))

    label = user_input.strip() or "[Image uploaded]"
    if image_bytes and user_input.strip():
        label += " [+ image]"
    st.session_state["turns"].append((label, turn))
    if image_bytes:
        # A new uploader key clears the photo so it is not re-sent with the next message.
        st.session_state["uploader_key"] += 1
    st.rerun()
