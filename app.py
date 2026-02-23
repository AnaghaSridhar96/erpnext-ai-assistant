import streamlit as st

st.set_page_config(page_title="ERPNext AI Assistant")

st.title("🤖 ERPNext AI Assistant")

st.write("This will become our ERPNext contextual AI chatbot.")

user_input = st.text_input("Ask a question about ERPNext:")

if user_input:
    st.write("You asked:", user_input)
    st.info("RAG system coming soon...")
