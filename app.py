import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import boto3
import base64
from dotenv import load_dotenv

# --- Custom Utils ---
from utils import *

# --- LangChain Community ---
from langchain_community.vectorstores import FAISS

# --- LangChain AWS ---
from langchain_aws import ChatBedrock, BedrockEmbeddings

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="GourmetGuide AI",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CREDENTIAL LOADING (THE FIX) ---
# 1. Load local .env file (for when running on your laptop)
load_dotenv()

# 2. Bridge Streamlit Cloud Secrets to System Environment
# Boto3 looks for keys in os.environ, but Streamlit Cloud stores them in st.secrets.
# We manually copy them over so Boto3 can find them.
if "AWS_ACCESS_KEY_ID" in st.secrets:
    os.environ["AWS_ACCESS_KEY_ID"] = st.secrets["AWS_ACCESS_KEY_ID"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = st.secrets["AWS_SECRET_ACCESS_KEY"]
    os.environ["AWS_DEFAULT_REGION"] = st.secrets["AWS_DEFAULT_REGION"]

# --- INITIALIZE AWS CLIENTS ---
try:
    # Boto3 will now automatically find the keys in os.environ
    bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

    # Define embeddings
    embeddings = BedrockEmbeddings(client=bedrock, model_id="amazon.titan-embed-text-v2:0")

    # Initialize LLM
    model_kwargs = {
        "max_tokens": 2048,
        "temperature": 0.0,
        "stop_sequences": ["\n\nHuman"],
    }
    llm = ChatBedrock(
        client=bedrock,
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        model_kwargs=model_kwargs,
    )
except Exception as e:
    st.error(f"Error initializing AWS clients: {e}")
    st.stop()

# --- LOAD FAISS INDEX ---
try:
    if os.path.exists("output/faiss_index"):
        db = FAISS.load_local(
            "output/faiss_index", embeddings, allow_dangerous_deserialization=True
        )
    else:
        st.error("FAISS index not found. Please run your ingestion script first.")
        st.stop()
except Exception as e:
    st.error(f"Error loading Vector Database: {e}")
    st.stop()

# --- SESSION STATE ---
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "assistant_response" not in st.session_state:
    st.session_state["assistant_response"] = []

# --- SIDEBAR UI ---
with st.sidebar:
    st.title("🍽️ Settings")
    st.markdown("---")
    
    st.subheader("📷 Visual Search")
    uploaded_image = st.file_uploader(
        "Upload a food image to find similar dishes:", 
        type=["png", "jpg", "jpeg"]
    )
    
    if uploaded_image:
        st.image(uploaded_image, caption="Preview", use_container_width=True)
    
    st.markdown("---")
    if st.button("🗑️ Clear Chat History", type="primary"):
        st.session_state["generated"] = []
        st.session_state["past"] = []
        st.session_state["assistant_response"] = []
        st.rerun()

# --- MAIN CHAT INTERFACE ---
st.title("GourmetGuide AI")
st.markdown("### *Your Personal Culinary Concierge*")
st.markdown("Ask me about cuisines, dietary preferences, or upload a photo of a dish you love!")
st.divider()

# Display Chat History
chat_container = st.container()

with chat_container:
    # Iterate through history (User and Assistant pairs)
    for i in range(len(st.session_state["generated"])):
        # 1. User Message
        with st.chat_message("user"):
            st.markdown(st.session_state["past"][i])
        
        # 2. Assistant Message
        response_data, images = st.session_state["generated"][i]
        
        with st.chat_message("assistant"):
            # Check if it's a recommendation list (list) or normal text (str)
            if isinstance(response_data, list):
                st.markdown("Here are some recommendations based on your taste:")
                for j, rec in enumerate(response_data):
                    # Create a card-like layout for each dish
                    with st.container():
                        col1, col2 = st.columns([1, 3])
                        
                        # Image Column
                        image_path = list(images.keys())[j]
                        metadata = images[image_path]
                        
                        with col1:
                            st.image("data/" + image_path, use_container_width=True)
                        
                        # Details Column
                        with col2:
                            st.subheader(metadata.get('menu_item_name', 'Unknown Dish'))
                            st.markdown(f"**{rec}**")
                            st.caption(f"📍 {metadata.get('restaurant_name', 'N/A')} | ⭐ {metadata.get('average_rating', 'N/A')}")
                            
                            # Metrics in a mini-row
                            m1, m2, m3 = st.columns(3)
                            m1.metric("Price", f"${metadata.get('price', '0')}")
                            m2.metric("Calories", metadata.get('calories', 'N/A'))
                            m3.metric("Serves", metadata.get('serves', '1'))
                        
                        st.divider()
            else:
                # Normal conversational response
                st.markdown(response_data)

# --- INPUT AREA ---
# We use a form so hitting 'Enter' works naturally
with st.form(key="chat_form", clear_on_submit=True):
    col_input, col_btn = st.columns([6, 1])
    
    with col_input:
        user_input = st.text_input(
            "Type your request...", 
            placeholder="E.g., I want something spicy and italian...",
            key="input_field"
        )
    
    with col_btn:
        # Align button with input box
        st.write("") 
        st.write("")
        submit_button = st.form_submit_button("Send 🚀")

# --- LOGIC HANDLER ---
if submit_button and (user_input or uploaded_image):
    original_input = user_input[:]
    image_flag = "no"

    # Handle Image Analysis
    if uploaded_image:
        image_flag = "yes"
        # Reset file pointer if needed, though Streamlit handles this well usually
        uploaded_image.seek(0)
        encoded_image = base64.b64encode(uploaded_image.read()).decode("utf-8")
        
        with st.spinner("Analyzing your image..."):
            image_description = describe_input_image(encoded_image, llm)
        
        # Enhance prompt with image context
        user_input = (
            f"I am looking for this dish, recommend similar dishes: {user_input} {image_description}"
        )

    # Save User Input to History
    st.session_state.past.append(original_input if original_input.strip() else "[Image Uploaded]")

    with st.spinner("Thinking..."):
        # Enhance search query
        enhanced_search_query = enhance_search(user_input, llm)
        enhanced_search_query = clean_text(enhanced_search_query)

        # Retrieve documents
        results = db.similarity_search(user_input, k=5)
        
        context = ""
        for doc in results:
            context += doc.page_content + "\n\n"

        # Prepare History for Memory
        chat_history = []
        if len(st.session_state["past"]) > 1:
            # Pair past inputs with past responses (excluding current turn)
            # We filter out complex objects (tuples) from history for the LLM context
            past_texts = st.session_state["past"][:-1]
            past_responses = st.session_state["assistant_response"]
            
            for u, a in zip(past_texts, past_responses):
                chat_history.append((u, a))

        # Generate Assistant Response
        chatbot_response_raw = assistant(context, user_input, chat_history, llm)

        # Parse JSON
        try:
            chatbot_response = json.loads(chatbot_response_raw)
        except json.JSONDecodeError:
            chatbot_response = {"recommendation": "no", "response": chatbot_response_raw}

        recommendation = chatbot_response.get("recommendation", "no")
        response_text = chatbot_response.get("response", "")
        
        # Save raw text response for memory
        st.session_state.assistant_response.append(response_text)

        # Final Logic: Recommendation vs Conversation
        if recommendation == "yes":
            # Pass original input if image was used, to avoid passing the long description text
            search_input = original_input if image_flag == "yes" else user_input
            
            rec_response, relevant_images = recommend_dishes_by_preference(
                results, search_input, llm
            )
            st.session_state["generated"].append((rec_response, relevant_images))
        else:
            st.session_state["generated"].append((response_text, []))

    # Rerun to update the chat container immediately
    st.rerun()
