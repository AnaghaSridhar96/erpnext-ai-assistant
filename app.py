import streamlit as st
import faiss
import pickle
from groq import Groq
from sentence_transformers import SentenceTransformer

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(page_title="ERPNext AI Assistant")
st.title("🤖 ERPNext AI Assistant (RAG Enabled)")

# -------------------------
# Load Groq API Key
# -------------------------
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("❌ GROQ_API_KEY not found in Streamlit Secrets.")
    st.stop()

client = Groq(api_key=groq_api_key)

# -------------------------
# Load FAISS Index
# -------------------------
@st.cache_resource
def load_vector_store():
    index = faiss.read_index("vector_store/erpnext.index")
    with open("vector_store/metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return index, metadata, model

index, metadata, embed_model = load_vector_store()

# -------------------------
# Retrieve Relevant Context
# -------------------------
def retrieve_context(query, k=3):
    query_embedding = embed_model.encode([query])
    distances, indices = index.search(query_embedding, k)
    results = [metadata[i] for i in indices[0]]
    return "\n\n".join(results)

# -------------------------
# Chat Interface
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask something about ERPNext..."):

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieve context from FAISS
    context = retrieve_context(prompt)

    system_prompt = f"""
You are an expert ERPNext consultant.

Use ONLY the following documentation context to answer.

If answer is not in context, say:
"I could not find this in the official ERPNext documentation."

Documentation Context:
{context}
"""

    with st.chat_message("assistant"):
        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
            )

            reply = response.choices[0].message.content
            st.markdown(reply)

            st.session_state.messages.append(
                {"role": "assistant", "content": reply}
            )

        except Exception as e:
            st.error(f"⚠️ Error: {e}")
