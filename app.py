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
    if "total_tokens" not in st.session_state:
        st.session_state.total_tokens = 0

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

def get_ollama_response(prompt: str) -> Iterator[tuple[str, dict]]:
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
            if json_response.get("done"):
                # Return the final stats
                yield "", json_response
            else:
                # Return the response chunk and empty stats
                yield json_response.get("response", ""), {}

def display_thinking_section(thinking: str):
    """Display thinking process in a collapsible section"""
    with st.expander("💭 Show thinking process"):
        st.markdown(thinking)

def calculate_energy_consumption(num_tokens: int) -> float:
    """Calculate energy consumption in gigajoules based on token usage"""
    power_watts = 35  # Power consumption under load in watts
    runtime_per_token = 0.02  # Runtime per token in seconds
    
    total_runtime = runtime_per_token * num_tokens  # Total runtime in seconds
    total_energy_joules = power_watts * total_runtime  # Energy in joules
    energy_gj = total_energy_joules / 1e9  # Convert joules to gigajoules
    return energy_gj

def display_token_counter():
    """Display token counter and energy consumption in the sidebar"""
    st.sidebar.markdown("### Usage Metrics")
    
    # Token counter
    st.sidebar.metric(
        "Total Tokens Used",
        f"{st.session_state.total_tokens:,}",
        help="Total number of tokens processed by the model"
    )
    
    # Energy consumption
    energy_gj = calculate_energy_consumption(st.session_state.total_tokens)
    st.sidebar.metric(
        "Energy Consumption",
        f"{energy_gj:.6f} GJ",
        help=(
            "Estimated energy consumption based on:\n"
            "• 35W power consumption\n"
            "• 0.02s processing time per token"
        )
    )
    
    # Add equivalent metrics for context
    if energy_gj > 0:
        energy_kwh = energy_gj * 277.778  # Convert GJ to kWh
        st.sidebar.caption(
            f"Equivalent to:\n"
            f"• {energy_kwh:.2f} kWh of electricity\n"
            f"• Running a 60W light bulb for {(energy_kwh/0.06):.1f} hours"
        )

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

    # Display token counter
    display_token_counter()

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
            for response_chunk, stats in get_ollama_response(prompt):
                if response_chunk:
                    full_response += response_chunk
                    
                    # Extract thinking and response
                    thinking, current_response = extract_thinking_and_response(full_response)
                    
                    # Display thinking section if found and not already displayed
                    if thinking and not thinking_displayed:
                        display_thinking_section(thinking)
                        thinking_displayed = True
                    
                    # Display the response without think tags
                    message_placeholder.write(current_response + "▌")
                elif stats:
                    # Update token count from final stats
                    if "prompt_eval_count" in stats:
                        st.session_state.total_tokens += stats["prompt_eval_count"]
                        display_token_counter()
            
            # Final update without cursor
            thinking, final_response = extract_thinking_and_response(full_response)
            if thinking and not thinking_displayed:
                display_thinking_section(thinking)
            message_placeholder.write(final_response)
            
        # Add assistant response to chat history (without think tags)
        st.session_state.messages.append({"role": "assistant", "content": final_response})

if __name__ == "__main__":
    main()
