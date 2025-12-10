import streamlit as st
from agent import Agent

st.set_page_config(page_title="Toyota Agent", layout="wide")

@st.cache_resource
def get_agent():
    return Agent()

agent = get_agent()

st.title("Toyota/Lexus Assistant")
st.markdown("ask questions about Sales (SQL) or Contracts/Manuals (RAG).")

# Example Buttons
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("Sales: RAV4 Germany"):
        st.session_state.query = "Monthly RAV4 HEV sales in Germany in 2024"
with col2:
    if st.button("RAG: Warranty Europe"):
        st.session_state.query = "What is the standard Toyota warranty for Europe?"
with col3:
    if st.button("RAG: child restraint system"):
        st.session_state.query = "Should I have any consideration When installing a child restraint system to a front passenger seat?"
with col4:
    if st.button("Hybrid: Sales vs Warranty"):
        st.session_state.query = "Compare Toyota vs Lexus SUV sales and summarize warranty differences."

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Handle button click or new input
query = st.chat_input("Ask a question...")
if "query" in st.session_state and st.session_state.query:
    query = st.session_state.query
    st.session_state.query = None # clear after use

if query:
    st.chat_message("user").write(query)
    
    with st.spinner("Thinking..."):
        result = agent.ask(query)
        
        with st.chat_message("assistant"):
            st.write(result["answer"])
            
            # Show Citations / Tool Usage
            with st.expander("Tool Usage & Sources"):
                st.write(f"**Tool Used:** {result['tool_used']}")
                
                if result.get("sql_query"):
                    st.code(result["sql_query"], language="sql")
                    
                if result.get("sources"):
                    st.write("**Sources:**")
                    for src in result["sources"]:
                        st.text(src)

