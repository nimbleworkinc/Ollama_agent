# Chat interface

# import the necessary libraries
import json
import requests
import streamlit as st
from typing import Iterator
import re

def init_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_started" not in st.session_state:
        st.session_state.conversation_started = False

def extract_thinking_and_response(text: str) -> tuple[str | None, str]:
    """Extract thinking process and response from the text"""
    # Find content within <think> tags
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    
    if think_match:
        thinking = think_match.group(1).strip()
        # Remove the think tags and content from the response
        response = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        return thinking, response
    
    return None, text

def get_ollama_response(prompt: str) -> Iterator[str]:
    """Get streaming response from Ollama API"""
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "deepseek-r1",
            "prompt": prompt,
            "stream": True
        },
        stream=True
    )
    
    for line in response.iter_lines():
        if line:
            json_response = json.loads(line)
            if not json_response.get("done"):
                yield json_response.get("response", "")

def display_thinking_section(thinking: str):
    """Display thinking process in a collapsible section"""
    with st.expander("💭 Show thinking process"):
        st.markdown(thinking)

def main():
    # Page config
    st.set_page_config(
        page_title="Chat with Deepseek",
        page_icon="🤖",
        layout="centered"
    )

    # Initialize session state
    init_session_state()

    # Custom CSS
    st.markdown("""
        <style>
        .user-avatar {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            background-color: #0068C9;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
        }
        .assistant-avatar {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            background-color: #09AB3B;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
        }
        .chat-message {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            display: flex;
            gap: 1rem;
        }
        .user-message {
            background-color: #F0F2F6;
        }
        .assistant-message {
            background-color: #FFFFFF;
        }
        </style>
    """, unsafe_allow_html=True)

    # Chat title
    st.title("💬 Chat with Deepseek")
    st.caption("Powered by Ollama")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if prompt := st.chat_input("Send a message"):
        # Add user message to chat
        st.chat_message("user").write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Get and display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            thinking_displayed = False
            
            # Stream the response
            for response_chunk in get_ollama_response(prompt):
                full_response += response_chunk
                
                # Extract thinking and response
                thinking, current_response = extract_thinking_and_response(full_response)
                
                # Display thinking section if found and not already displayed
                if thinking and not thinking_displayed:
                    display_thinking_section(thinking)
                    thinking_displayed = True
                
                # Display the response without think tags
                message_placeholder.write(current_response + "▌")
            
            # Final update without cursor
            thinking, final_response = extract_thinking_and_response(full_response)
            if thinking and not thinking_displayed:
                display_thinking_section(thinking)
            message_placeholder.write(final_response)
            
        # Add assistant response to chat history (without think tags)
        st.session_state.messages.append({"role": "assistant", "content": final_response})

if __name__ == "__main__":
    main()
