import streamlit as st
from backend.run_llm import run_llm


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


prompt = st.chat_input("Ask anything...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.spinner("Wait for it...", show_time=True):
        answer = run_llm(prompt)
        urls = []
        for doc in answer["context"]:
            urls.append(doc.metadata["source"])
        
        full_answer = f"{answer['answer']} \n\n --Resources-- \n\n\n\n {"\n\n".join(urls)}"
        st.session_state.messages.append({"role": "assistant", "content": full_answer})
        with st.chat_message("assistant"):
            st.markdown(full_answer)
