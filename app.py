import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

st.set_page_config(
    page_title="LegalEase AI",
    page_icon="⚖️",
    layout="wide",
)

SYSTEM_PROMPT = """
You are LegalEase AI, an AI assistant focused on explaining legal concepts
in simple, easy-to-understand language.

You can:
- Explain legal terminology
- Explain contracts
- Summarize legal documents
- Explain basic rights
- Generate questions for a lawyer
- Explain legal procedures
- Classify legal documents

Rules:
1. Explain complicated legal concepts in plain language.
2. Define legal terminology when it appears.
3. Give examples when useful.
4. Do not pretend to be a lawyer.
5. Do not provide definitive legal advice.
6. For important legal decisions, recommend consulting a qualified lawyer.
7. If the answer depends on jurisdiction, ask which country/state/jurisdiction.
8. Be clear about uncertainty and avoid making up laws or citations.
"""

if "messages" not in st.session_state:
    st.session_state.messages = [SystemMessage(content=SYSTEM_PROMPT)]

if "connected" not in st.session_state:
    st.session_state.connected = False

st.title("⚖️ LegalEase AI")
st.caption("Understand the law in simple language.")

with st.sidebar:
    st.header("⚙️ AI Connection")

    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-..."
    )

    model = st.selectbox(
        "Model",
        ["gpt-4o-mini", "gpt-4o"],
        index=0,
    )

    if st.button("🔌 Connect", use_container_width=True):
        if not api_key.strip():
            st.error("Please enter your OpenAI API key.")
        else:
            st.session_state.api_key = api_key.strip()
            st.session_state.model = model
            st.session_state.connected = True
            st.success("Connected.")

    if st.session_state.connected:
        st.caption("● OpenAI connected")

    st.divider()
    st.header("💬 Ask LegalEase")

    prompt = st.text_area(
        "Your legal question",
        placeholder=(
            "Example:\n"
            "What does indemnification mean in a contract?"
        ),
        height=150,
    )

    if st.button("⚖️ Ask LegalEase", use_container_width=True):
        if not st.session_state.connected:
            st.warning("Connect your OpenAI API key first.")
        elif not prompt.strip():
            st.warning("Enter a legal question first.")
        else:
            st.session_state.messages.append(
                HumanMessage(content=prompt.strip())
            )
            try:
                chat = ChatOpenAI(
                    model=st.session_state.model,
                    temperature=0.3,
                    api_key=st.session_state.api_key,
                )
                with st.spinner("LegalEase is thinking..."):
                    response = chat.invoke(st.session_state.messages)

                st.session_state.messages.append(
                    AIMessage(content=response.content)
                )
            except Exception as exc:
                st.error(f"Request failed: {exc}")

    st.divider()
    st.caption(
        "LegalEase AI provides general legal information, "
        "not legal advice."
    )

if len(st.session_state.messages) == 1:
    st.subheader("What can I help you understand?")
    cols = st.columns(4)
    quick_actions = [
        ("⚖️", "Legal Terms", "What does indemnification mean?"),
        ("📄", "Contracts", "Explain the important parts of a contract."),
        ("📝", "Summarize", "How should I summarize a legal document?"),
        ("🛡️", "Basic Rights", "Explain my basic rights in simple language."),
    ]

    for col, (icon, title, example) in zip(cols, quick_actions):
        with col:
            st.markdown(f"### {icon} {title}")
            st.caption(example)

    st.info(
        "Tip: Include the country/state when asking about a law or legal procedure."
    )

for message in st.session_state.messages[1:]:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)
