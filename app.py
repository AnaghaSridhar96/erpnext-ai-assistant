import streamlit as st
from groq import Groq

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(page_title="ERPNext AI Assistant")
st.title("🤖 ERPNext AI Assistant")
st.write("ERPNext contextual AI chatbot (Groq + Llama 3)")

# -------------------------
# Load API Key Securely
# -------------------------
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("❌ GROQ_API_KEY not found in Streamlit Secrets.")
    st.stop()

client = Groq(api_key=groq_api_key)

# -------------------------
# Initialize Chat History
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": """
You are an expert ERPNext and Frappe consultant.

Your job:
- Provide accurate ERPNext guidance
- Give step-by-step instructions
- Focus on configuration and customization
- Suggest best practices
- Avoid hallucinating features that don't exist
- If unsure, say you are not certain

Answer in structured, professional format.
"""
        }
    ]

# -------------------------
# Display Previous Messages
# -------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -------------------------
# Chat Input
# -------------------------
if prompt := st.chat_input("Ask something about ERPNext..."):

    # Add user message to history
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Assistant Response
    with st.chat_message("assistant"):

        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=st.session_state.messages,
            )

            reply = response.choices[0].message.content
            st.markdown(reply)

            # Save assistant response
            st.session_state.messages.append(
                {"role": "assistant", "content": reply}
            )

        except Exception as e:
            st.error(f"⚠️ Error from Groq API: {e}")
