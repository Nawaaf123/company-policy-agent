import streamlit as st
from app import ask_question

st.set_page_config(
    page_title="MRFOG AI ASSISTANT",
    page_icon="🤖",
    layout="centered"
)

# -----------------------------
# Header / Logo
# -----------------------------
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.image("assets/mrfog_logo.png", width=220)

st.markdown(
    "<h1 style='text-align: center;'>MRFOG AI ASSISTANT</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center;'>Ask questions about MR FOG products, categories, specifications, support, warranty, returns, distributors, and website information.</p>",
    unsafe_allow_html=True
)

st.divider()

# -----------------------------
# Chat History
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# -----------------------------
# Chat Input
# -----------------------------
user_question = st.chat_input("Ask about MR FOG products, categories, support, warranty, or distributor info...")

if user_question:
    st.session_state.messages.append({
        "role": "user",
        "content": user_question
    })

    with st.chat_message("user"):
        st.write(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Searching MR FOG knowledge base..."):
            result = ask_question(user_question)

            answer = result["answer"]
            grounded = result["grounded"]

            st.write(answer)

            if grounded:
                st.success("Grounded answer from MR FOG documents")
            else:
                st.warning("Answer not found in MR FOG documents")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.image("assets/mrfog_logo.png", width=160)
    st.markdown("### MRFOG AI ASSISTANT")
    st.write("This assistant answers questions using MR FOG website/product documents.")

    st.markdown("### Example Questions")
    st.write("- How many product categories does MR FOG have?")
    st.write("- What is NOVA?")
    st.write("- What is SWITCH POD 45K?")
    st.write("- What is DRT?")
    st.write("- How can someone become a distributor?")
    st.write("- What warranty or return information is available?")