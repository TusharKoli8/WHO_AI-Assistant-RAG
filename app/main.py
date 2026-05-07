import streamlit as st
from agents import run_agent_pipeline
from rag_chain import generate_answer

st.set_page_config(
    page_title="WHO_AI Assistant",
    page_icon="🏥",
    layout="centered"
)

st.title("WHO_AI Assistant")
st.caption("Your intelligent Q&A Assistant powered by WHO Publications")


if "messages" not in st.session_state:
    st.session_state.messages = []

## Sidebar

with st.sidebar:
    st.title("Chat History")
    st.markdown("### WHO_AI")
    st.caption("Answers sourced from WHO Publications only.")
    st.divider()
     
    if not st.session_state.messages:
        st.write("No chat history available.")
    else:

        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.text(f"{msg['content'][:30]}...") 
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()               


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


query = st.chat_input("Ask anything from WHO Publications...")

if query:

    with st.chat_message("user"):
        st.markdown(query)

    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("assistant"):

        with st.spinner("Analyzing..."):
            agent_result = run_agent_pipeline(query)

        if not agent_result['is_valid']:
            response = f" **Your question is outside the scope of WHO Publications.**\n\n_{agent_result['reject_reason']}_\n\nPlease ask something related to the WHO documents loaded in the system."
            st.markdown(response)

        else:
            with st.expander(" Query Analysis", expanded=False):
                st.markdown(f"**Category:** `{agent_result['category']}`")
                st.markdown(f"**Refined:** _{agent_result['refined_query']}_")

            with st.spinner(" Searching WHO documents..."):
                result = generate_answer(agent_result['refined_query'])

            
     #       sources_md = ""
     #       if result['sources']:
     #           lines = "\n".join([f"- `{s['file']}` — Page **{s['page']}**" for s in result['sources']])
     #           sources_md = f"\n\n---\n** Sources:**\n{lines}"

            response = result['answer']# + sources_md

            st.markdown(response)

   
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    st.rerun()

