
# 🍽️ GourmetGuide AI: Multimodal RAG Food Recommendation System

> **A context-aware recommendation engine powered by AWS Bedrock, Claude 3, and Vector Search.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![AWS](https://img.shields.io/badge/AWS-Bedrock-orange)](https://aws.amazon.com/bedrock/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-v0.1-green)](https://langchain.com/)

## 📖 Project Objective
The goal of this project is to solve the "choice paralysis" problem in food ordering by building a **Multimodal Retrieval-Augmented Generation (RAG)** system. Unlike traditional keyword search, GourmetGuide AI understands both **textual preferences** ("I want something spicy and Italian") and **visual inputs** (uploading a photo of a dish) to generate personalized, context-aware recommendations from a curated restaurant dataset.

## ⚙️ High-Level Architecture
The system follows a modern RAG pipeline deployed on AWS infrastructure:

1.  **Data Ingestion:** Restaurant menu data (images & metadata) is stored in **Amazon S3**.
2.  **Multimodal Processing:**
    * **Text:** Embedded using **Amazon Titan Embeddings v2**.
    * **Images:** Analyzed using **Claude 3 Sonnet** to generate descriptive textual representations.
3.  **Vector Storage:** Embeddings are indexed in **FAISS** for low-latency similarity retrieval.
4.  **Retrieval & Generation:**
    * User query (Text/Image) -> Vector Search -> Top-K Relevant Contexts.
    * **Claude 3 Sonnet** synthesizes the context + conversation history to generate a natural language response.
5.  **Frontend:** An interactive **Streamlit** application allows users to chat and upload images in real-time.



## 🚀 Key Features
* **📷 Visual Search (Multimodal):** Users can upload an image of food (e.g., a photo from Instagram), and the system uses **Claude 3 Vision** to interpret the dish and find similar items in the menu.
* **🧠 Context-Aware Memory:** Maintains conversational state (up to 5 turns), allowing users to refine requests (e.g., "Show me pasta" -> "What about something cheaper?").
* **🔍 Hybrid Search:** Combines semantic vector search (FAISS) with structured filtering (Dietary preferences, Price, Cuisine).
* **☁️ Cloud-Native:** Fully integrated with **AWS Bedrock** via `boto3` for scalable model inference.

## 🛠️ Tech Stack
* **Language:** Python 3.10+
* **GenAI Models:** Anthropic Claude 3 Sonnet (Reasoning & Vision), Amazon Titan v2 (Embeddings).
* **Orchestration:** LangChain (Chains, Prompts, Memory).
* **Vector Database:** FAISS (Facebook AI Similarity Search).
* **Cloud Infrastructure:** AWS S3 (Storage), AWS Bedrock (Model-as-a-Service).
* **Frontend:** Streamlit & Streamlit Chat.

## 📂 Project Structure
```bash
├── app.py                 # Main Streamlit application entry point
├── utils.py               # Core logic (RAG pipeline, Bedrock client, Search)
├── admin_utils.py         # Data ingestion and vector DB creation scripts
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (AWS Keys - Not committed)
└── output/
    └── faiss_index/       # Local vector store indices
````

## ⚡ Setup & Installation

**1. Clone the Repository**

```bash
git clone [https://github.com/YOUR_USERNAME/food-recommendation-rag.git](https://github.com/YOUR_USERNAME/food-recommendation-rag.git)
cd food-recommendation-rag
```

**2. Create Virtual Environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**3. Install Dependencies**

```bash
pip install -r requirements.txt
```

**4. Configure Credentials**
Create a `.env` file in the root directory:

```ini
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-east-1
```

*Note: Ensure your AWS account has model access enabled for Claude 3 and Titan Embeddings in the us-east-1 region.*

**5. Run the Application**

```bash
streamlit run app.py
```

## 📊 Sample Interaction

> **User:** *Uploads a picture of a pepperoni pizza*
>
> **System:** "This looks like a classic Pepperoni Pizza with a crispy crust. Based on this, I recommend the 'Spicy Salami Pizza' from Tony's Bistro. It has similar toppings but with an extra kick of chili oil. Would you like to check the nutrition facts?"
<img width="1440" height="802" alt="image" src="https://github.com/user-attachments/assets/61846b9c-8627-4b7a-8ceb-200dff5018d9" />

-----

Developed by Aman Singh 
```
```
