"""
app.py
======
A clean, interactive Streamlit chat interface for the College Helpdesk Chatbot.

Key Features:
- 100% Pure Python (No HTML, CSS, or JavaScript).
- Uses st.chat_input and st.chat_message for a conversational interface.
- Preserves conversation history across reruns using st.session_state.
- Defensive error handling: alerts user if model artifacts are missing.
"""

import streamlit as st
import os
import sys

# Ensure local imports work regardless of execution directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chatbot import CollegeChatbot, CONFIDENCE_THRESHOLD

# ------------------------------------------------------------------------------
# 1. Page Configuration
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="College Helpdesk AI Chatbot",
    page_icon="🎓",
    layout="centered"
)

# ------------------------------------------------------------------------------
# 2. Chatbot Instance Loader (Cached to avoid reloading on every interaction)
# ------------------------------------------------------------------------------
@st.cache_resource
def load_chatbot():
    """
    Loads and caches the trained CollegeChatbot model instance.
    If model files are missing, returns None so app.py can display a helpful error.
    """
    try:
        return CollegeChatbot()
    except FileNotFoundError as err:
        return str(err)
    except Exception as exc:
        return f"Unexpected error while loading chatbot: {exc}"


# ------------------------------------------------------------------------------
# 3. Sidebar: Helpful Information & Quick Guide
# ------------------------------------------------------------------------------
with st.sidebar:
    st.header("🎓 College Helpdesk")
    st.write(
        "Welcome! I am **CampusBot**, your automated college helpdesk assistant. "
        "Ask me anything about admissions, courses, fees, or campus hours!"
    )
    st.divider()

    st.subheader("💡 Topics I Can Answer:")
    st.markdown(
        """
        - **Greetings & Help:** "Hello", "How can you help?"
        - **College Timings:** "When is the library open?"
        - **Offered Courses:** "What engineering branches do you offer?"
        - **Tuition & Fees:** "How much are the semester fees?"
        - **Campus Contact:** "What is the admissions phone number?"
        """
    )
    st.divider()

    # Clear chat history button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am CampusBot, your college helpdesk assistant. How can I help you today?"}
        ]
        st.rerun()

# ------------------------------------------------------------------------------
# 4. Main Chat Interface
# ------------------------------------------------------------------------------
st.title("🎓 College Helpdesk Assistant")
st.caption("Powered by Natural Language Processing (TF-IDF & Logistic Regression)")

# Check if model loaded successfully
bot_or_error = load_chatbot()

if isinstance(bot_or_error, str):
    # Model files missing or load error
    st.error("⚠️ Model files not found!")
    st.warning(bot_or_error)
    st.info("💡 Solution: Run the training script first:\n\n```powershell\npython train.py\n```")
    st.stop()

bot = bot_or_error

# ------------------------------------------------------------------------------
# 5. Initialize Chat History in st.session_state
# WHY session_state: Streamlit reruns the script from top to bottom on each interaction.
# st.session_state keeps conversation history persistent between user messages.
# ------------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am CampusBot, your college helpdesk assistant. How can I help you today?"}
    ]

# Display all existing chat messages from history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ------------------------------------------------------------------------------
# 6. User Message Input & Response Generation
# ------------------------------------------------------------------------------
user_query = st.chat_input("Ask a question about timings, fees, courses, or contact...")

if user_query:
    # 1. Display user query in the chat container
    with st.chat_message("user"):
        st.write(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    # 2. Generate response from the chatbot
    response_text = bot.get_response(user_query)

    # 3. Display bot response in the chat container
    with st.chat_message("assistant"):
        st.write(response_text)
    st.session_state.messages.append({"role": "assistant", "content": response_text})
