import streamlit as st
from app import ask_question

st.set_page_config(
    page_title="MRFOG AI ASSISTANT",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 MRFOG AI ASSISTANT")
st.write("Ask questions about MR FOG products, categories, specifications, support, warranty, returns, distributors, and website information.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_question = st.chat_input("Ask a company policy question...")

if user_question:
    st.session_state.messages.append({
        "role": "user",
        "content": user_question
    })

    with st.chat_message("user"):
        st.write(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Searching company documents..."):
            result = ask_question(user_question)

            answer = result["answer"]
            grounded = result["grounded"]

            st.write(answer)

            if grounded:
                st.success("Grounded answer from company documents")
            else:
                st.warning("Answer not found in company documents")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })