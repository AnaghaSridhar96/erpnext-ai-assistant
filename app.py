import streamlit as st
from groq import Groq

st.set_page_config(page_title="ERPNext AI Assistant")
st.title("🤖 ERPNext AI Assistant")
st.write("ERPNext contextual AI chatbot (Groq + Llama 3)")

# Load API key from Streamlit secrets
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask something about ERPNext..."):

    # Add user message
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):

        response = client.chat.completions.create(
            model="llama3-8b-8192",   # fast + free
            messages=st.session_state.messages,
        )

        reply = response.choices[0].message.content
        st.markdown(reply)

    # Save assistant response
    st.session_state.messages.append(
        {"role": "assistant", "content": reply}
    )
