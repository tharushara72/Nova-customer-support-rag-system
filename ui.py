import streamlit as st
import requests

st.set_page_config(
    page_title="Nova - Customer Support",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Nova - Customer Support Assistant")
st.caption("Powered by RAG — Retrieval Augmented Generation")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("latency"):
            st.caption(f"⚡ {msg['latency']}ms  |  retrieval: {msg['stages']['retrieval_ms']}ms  |  rerank: {msg['stages']['reranking_ms']}ms  |  generation: {msg['stages']['generation_ms']}ms")

# Input box at bottom
if query := st.chat_input("Ask a customer support question..."):

    # Show user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    # Call the RAG API
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    "http://localhost:8000/ask",
                    json={"query": query},
                    timeout=30
                )
                data = response.json()

                st.write(data["response"])
                st.caption(
                    f"⚡ {data['total_latency_ms']}ms  |  "
                    f"retrieval: {data['stage_breakdown']['retrieval_ms']}ms  |  "
                    f"rerank: {data['stage_breakdown']['reranking_ms']}ms  |  "
                    f"generation: {data['stage_breakdown']['generation_ms']}ms"
                )

                # Show sources in expander
                with st.expander("📚 Retrieved sources"):
                    for i, source in enumerate(data["sources"], 1):
                        st.write(f"**{i}.** {source}")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": data["response"],
                    "latency": data["total_latency_ms"],
                    "stages": data["stage_breakdown"]
                })

            except Exception as e:
                st.error(f"Error: {str(e)}. Make sure the API is running on port 8000.")