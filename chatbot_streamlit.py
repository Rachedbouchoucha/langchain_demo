import streamlit as st
import os
from dotenv import load_dotenv

from agents_main import get_response


@st.cache_resource(show_spinner=False)
def get_lambda_url():
    load_dotenv(dotenv_path='.env')
    MY_ENV_VAR = os.getenv('LambdaUrl')
    return MY_ENV_VAR


if "lambda_url" not in st.session_state.keys():
    st.session_state.lambda_url = get_lambda_url()

st.set_page_config(page_title="LlamaIndex Test", layout="centered",
                   initial_sidebar_state="auto", menu_items=None)
st.title("ChatBot Test")

if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me any question!"}
    ]

if prompt := st.chat_input("Your question"):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:  # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = get_response(prompt)
            st.write(response)
            message = {"role": "assistant", "content": response}
            st.session_state.messages.append(message)  # Add response to message history
