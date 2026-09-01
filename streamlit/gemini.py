import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load .env
load_dotenv()


# -----------------------------
# Gemini model
# -----------------------------
def get_model_from_gcp(
    model_name: str = "gemini-2.5-flash-lite",
) -> ChatGoogleGenerativeAI:

    return ChatGoogleGenerativeAI(
        model=model_name,
    )


# -----------------------------
# Streamlit page
# -----------------------------
st.set_page_config(
    page_title="Gemini Assistant",
    page_icon="🤖",
)

st.title("🤖 Gemini Assistant")
st.caption("Ask anything to Gemini")


# -----------------------------
# Chat history
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# -----------------------------
# Display previous messages
# -----------------------------
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# -----------------------------
# User input
# -----------------------------
user_question = st.chat_input("Ask Gemini anything...")


if user_question:

    # Display user question
    with st.chat_message("user"):
        st.markdown(user_question)

    # Save user question
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question,
        }
    )

    # Get Gemini model
    model = get_model_from_gcp()

    # Ask Gemini
    with st.chat_message("assistant"):

        with st.spinner("Gemini is thinking..."):

            response = model.invoke(user_question)

            answer = response.content

        st.markdown(answer)

    # Save Gemini response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )