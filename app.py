import streamlit as st

from openai import OpenAI
from openai import AuthenticationError, APIConnectionError, APIError, RateLimitError

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="LegalEase AI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# LEGAL AI SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are LegalEase AI, a legal-information assistant.

Your job is to explain legal concepts in simple, plain language.

You can help with:

- Explain legal terminology
- Explain contracts and clauses
- Summarize legal documents
- Explain basic rights
- Generate questions for a lawyer
- Explain legal procedures
- Classify legal documents

Important rules:

1. Use simple language.
2. Explain legal jargon whenever possible.
3. Give examples when useful.
4. Do not claim to be a lawyer.
5. Do not present your response as a substitute for legal advice.
6. Laws differ between countries, states and jurisdictions.
7. If jurisdiction matters, ask the user for their country/state.
8. Never invent laws, statutes, cases, deadlines or citations.
9. For important legal decisions, recommend consulting a qualified lawyer.
10. Clearly state uncertainty when the available information is insufficient.
"""


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "api_key" not in st.session_state:
    st.session_state.api_key = ""

if "model" not in st.session_state:
    st.session_state.model = "gpt-4o-mini"

if "connected" not in st.session_state:
    st.session_state.connected = False

if "connection_error" not in st.session_state:
    st.session_state.connection_error = ""


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

@import url(
'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700'
'&family=Playfair+Display:wght@600;700&display=swap'
);


/* ==========================================================
   GLOBAL
   ========================================================== */

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #f7f8fa;
    color: #182230;
}

.block-container {
    max-width: 1120px;
    padding-top: 2rem;
    padding-bottom: 7rem;
}


/* ==========================================================
   SIDEBAR
   ========================================================== */

section[data-testid="stSidebar"] {
    background: #eef1f5;
    border-right: 1px solid #e2e5ea;
}

section[data-testid="stSidebar"] > div {
    padding: 1.5rem 1.25rem;
}


/* Sidebar headings */

section[data-testid="stSidebar"] h3 {
    color: #182230;
    font-size: 17px;
}


/* Sidebar labels */

section[data-testid="stSidebar"] label {
    color: #344054;
    font-size: 13px;
}


/* Sidebar brand */

.brand {
    padding: 8px 4px 22px 4px;
}

.brand-mark {
    width: 44px;
    height: 44px;
    border-radius: 13px;

    background: #182230;
    color: #ffffff;

    display: flex;
    align-items: center;
    justify-content: center;

    font-size: 22px;

    margin-bottom: 12px;
}

.brand-name {
    font-family: 'Playfair Display', serif;

    font-size: 25px;
    font-weight: 700;

    color: #182230;
}

.brand-sub {
    color: #6b7482;

    font-size: 11px;

    margin-top: 3px;
}


/* Sidebar section labels */

.sidebar-label {
    color: #929aa6;

    font-size: 10px;
    font-weight: 700;

    letter-spacing: 1.2px;

    text-transform: uppercase;

    margin-top: 18px;
    margin-bottom: 8px;
}


/* Connection status */

.connection {
    border: 1px solid #dce8df;

    background: #f8fcf9;

    border-radius: 12px;

    padding: 10px 12px;

    font-size: 12px;

    color: #52605a;

    margin-top: 10px;
}

.dot {
    display: inline-block;

    width: 7px;
    height: 7px;

    background: #45a36b;

    border-radius: 50%;

    margin-right: 7px;
}


/* Error box */

.api-error {
    background: #fff6f5;

    border: 1px solid #f2c9c5;

    color: #9b332a;

    border-radius: 12px;

    padding: 12px;

    margin-top: 10px;

    font-size: 12px;

    line-height: 1.5;
}


/* ==========================================================
   MAIN HERO
   ========================================================== */

.hero {
    text-align: center;

    padding: 45px 20px 20px;
}

.hero-kicker {
    color: #a27f4e;

    font-size: 11px;

    font-weight: 700;

    letter-spacing: 1.8px;

    text-transform: uppercase;

    margin-bottom: 13px;
}

.hero h1 {
    font-family: 'Playfair Display', serif;

    font-size: 48px;

    line-height: 1.08;

    letter-spacing: -1.5px;

    margin: 0;

    color: #182230;
}

.hero p {
    max-width: 600px;

    margin: 15px auto 0;

    color: #6c7786;

    font-size: 15px;

    line-height: 1.65;
}


/* ==========================================================
   QUESTION BOX
   ========================================================== */

.question-card {
    background: #ffffff;

    border: 1px solid #e3e6eb;

    border-radius: 18px;

    padding: 18px;

    margin: 20px auto 28px;

    max-width: 820px;

    box-shadow:
        0 10px 35px rgba(20, 30, 45, 0.05);
}

.question-label {
    color: #182230;

    font-size: 13px;

    font-weight: 700;

    margin-bottom: 8px;
}


/* Text area */

div[data-testid="stTextArea"] textarea {
    border: 1px solid #e0e4e9;

    border-radius: 12px;

    background: #fbfcfd;

    color: #182230;

    font-size: 14px;

    padding: 14px;

    min-height: 105px;
}

div[data-testid="stTextArea"] textarea:focus {
    border-color: #b08d57;

    box-shadow: 0 0 0 1px #b08d57;
}


/* ==========================================================
   BUTTONS
   ========================================================== */

.stButton > button {
    border-radius: 10px;

    font-weight: 600;

    min-height: 42px;
}

.stButton > button[kind="primary"] {
    background: #182230;

    border-color: #182230;

    color: white;
}

.stButton > button[kind="primary"]:hover {
    background: #263447;

    border-color: #263447;
}


/* ==========================================================
   WELCOME / FEATURE CARDS
   ========================================================== */

.intro-card {
    background: white;

    border: 1px solid #e4e7ec;

    border-radius: 17px;

    padding: 20px 22px;

    margin: 10px auto 22px;

    max-width: 820px;

    box-shadow:
        0 7px 25px rgba(20, 30, 45, .03);
}

.intro-title {
    font-size: 15px;

    font-weight: 700;

    margin-bottom: 5px;
}

.intro-text {
    color: #687385;

    font-size: 13px;
}


.feature {
    background: white;

    border: 1px solid #e4e7ec;

    border-radius: 16px;

    padding: 18px;

    min-height: 135px;

    margin-bottom: 14px;

    box-shadow:
        0 7px 22px rgba(20, 30, 45, .03);
}

.feature-icon {
    font-size: 22px;

    margin-bottom: 11px;
}

.feature-title {
    font-size: 14px;

    font-weight: 700;

    margin-bottom: 6px;
}

.feature-text {
    color: #687385;

    font-size: 12px;

    line-height: 1.55;
}


/* ==========================================================
   CHAT
   ========================================================== */

div[data-testid="stChatMessage"] {
    border: 1px solid #e4e7ec;

    border-radius: 15px;

    padding: 10px 14px;

    background: white;

    margin-bottom: 10px;
}


/* ==========================================================
   DISCLAIMER
   ========================================================== */

.disclaimer {
    border-top: 1px solid #e4e7ec;

    margin-top: 25px;

    padding-top: 14px;

    color: #8b94a1;

    font-size: 11px;

    text-align: center;
}


/* ==========================================================
   MOBILE
   ========================================================== */

@media (max-width: 700px) {

    .hero {
        padding-top: 25px;
    }

    .hero h1 {
        font-size: 38px;
    }

}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="brand">

            <div class="brand-mark">
                ⚖
            </div>

            <div class="brand-name">
                LegalEase AI
            </div>

            <div class="brand-sub">
                Plain-language legal information
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-label">AI Connection</div>',
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # API KEY
    # --------------------------------------------------------

    api_key_input = st.text_input(
        "OpenAI API Key",
        value=st.session_state.api_key,
        type="password",
        placeholder="sk-...",
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    available_models = [
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4.1-mini",
        "gpt-4.1",
    ]

    model_input = st.selectbox(
        "Model",
        available_models,
        index=available_models.index(
            st.session_state.model
        ),
    )

    st.caption(
        "The selected model will be checked when you connect."
    )

    # --------------------------------------------------------
    # CONNECT BUTTON
    # --------------------------------------------------------

    if st.button(
        "Connect AI",
        type="primary",
        use_container_width=True,
    ):

        # Clear old status
        st.session_state.connection_error = ""
        st.session_state.connected = False

        if not api_key_input.strip():

            st.session_state.connection_error = (
                "Please enter your OpenAI API key."
            )

        else:

            try:

                # ------------------------------------------------
                # 1. CHECK API KEY
                # ------------------------------------------------

                client = OpenAI(
                    api_key=api_key_input.strip()
                )

                # models.list() requires valid authentication
                client.models.list()

                # ------------------------------------------------
                # 2. CHECK SELECTED MODEL
                # ------------------------------------------------

                client.models.retrieve(
                    model_input
                )

                # ------------------------------------------------
                # 3. EVERYTHING IS VALID
                # ------------------------------------------------

                st.session_state.api_key = api_key_input.strip()

                st.session_state.model = model_input

                st.session_state.connected = True

                st.session_state.connection_error = ""

            except AuthenticationError:

                st.session_state.connection_error = (
                    "Invalid OpenAI API key. "
                    "Please check the key and try again."
                )

            except RateLimitError:

                st.session_state.connection_error = (
                    "OpenAI rejected the request because of "
                    "a rate limit or account/billing restriction."
                )

            except APIConnectionError:

                st.session_state.connection_error = (
                    "Could not connect to OpenAI. "
                    "Check your internet connection and try again."
                )

            except APIError as error:

                st.session_state.connection_error = (
                    f"OpenAI API error: {error}"
                )

            except Exception as error:

                st.session_state.connection_error = (
                    f"Connection failed: {error}"
                )

    # --------------------------------------------------------
    # SHOW CONNECTION ERROR
    # --------------------------------------------------------

    if st.session_state.connection_error:

        st.markdown(
            f"""
            <div class="api-error">
                <strong>Connection failed</strong><br>
                {st.session_state.connection_error}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------
    # SHOW CONNECTED STATE
    # --------------------------------------------------------

    if st.session_state.connected:

        st.markdown(
            """
            <div class="connection">
                <span class="dot"></span>
                OpenAI connected successfully
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # ========================================================
    # LEGAL ASSISTANT
    # ========================================================

    st.markdown(
        '<div class="sidebar-label">Legal Assistant</div>',
        unsafe_allow_html=True,
    )

    assistant_mode = st.selectbox(
        "Focus",
        [
            "General Legal Information",
            "Contract Explanation",
            "Legal Document Summary",
            "Legal Terminology",
            "Basic Rights",
            "Legal Procedure",
            "Questions for a Lawyer",
            "Document Classification",
        ],
    )

    st.divider()

    # ========================================================
    # CONVERSATION
    # ========================================================

    st.markdown(
        '<div class="sidebar-label">Conversation</div>',
        unsafe_allow_html=True,
    )

    if st.button(
        "＋ New conversation",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.rerun()

    st.markdown("---")

    st.caption(
        "LegalEase AI"
    )

    st.caption(
        "General information only • Not legal advice"
    )


# ============================================================
# MAIN HERO
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-kicker">
            LEGAL INFORMATION, SIMPLIFIED
        </div>

        <h1>
            Understand the law.<br>
            Without the jargon.
        </h1>

        <p>
            Ask questions about legal terms, contracts,
            procedures and basic rights — and get clear
            explanations in everyday language.
        </p>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MAIN QUESTION BOX
# ============================================================

st.markdown(
    """
    <div class="question-card">

        <div class="question-label">
            Ask LegalEase
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# The actual input is in the MAIN AREA, not sidebar.

user_question = st.text_area(
    "Your legal question",
    placeholder=(
        "Example: What can my employer terminate me for "
        "without notice?"
    ),
    height=110,
    label_visibility="collapsed",
)


ask_col1, ask_col2, ask_col3 = st.columns([1, 1, 1])

with ask_col2:

    ask_button = st.button(
        "⚖️  Ask LegalEase",
        type="primary",
        use_container_width=True,
    )


# ============================================================
# WELCOME CARDS
# ============================================================

if not st.session_state.messages:

    st.markdown(
        """
        <div class="intro-card">

            <div class="intro-title">
                Welcome to LegalEase
            </div>

            <div class="intro-text">
                Ask a question about a legal term, contract,
                procedure or basic right. LegalEase will explain
                it in simple language.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    cards = [
        (
            "§",
            "Legal terminology",
            "Understand unfamiliar legal words and phrases.",
        ),
        (
            "▤",
            "Explain contracts",
            "Break down clauses and explain what they mean.",
        ),
        (
            "≡",
            "Summarize documents",
            "Turn lengthy legal text into a clear summary.",
        ),
        (
            "◈",
            "Basic rights",
            "Learn general information about common legal rights.",
        ),
        (
            "?",
            "Questions for a lawyer",
            "Prepare focused questions before a consultation.",
        ),
        (
            "→",
            "Legal procedures",
            "Understand a legal process step by step.",
        ),
    ]

    for start in range(0, len(cards), 3):

        row = cards[start:start + 3]

        columns = st.columns(3)

        for column, card in zip(columns, row):

            icon, title, description = card

            with column:

                st.markdown(
                    f"""
                    <div class="feature">

                        <div class="feature-icon">
                            {icon}
                        </div>

                        <div class="feature-title">
                            {title}
                        </div>

                        <div class="feature-text">
                            {description}
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# ============================================================
# DISPLAY PREVIOUS CHAT
# ============================================================

for message in st.session_state.messages:

    if isinstance(message, HumanMessage):

        with st.chat_message("user"):

            st.write(
                message.content
            )

    elif isinstance(message, AIMessage):

        with st.chat_message("assistant"):

            st.write(
                message.content
            )


# ============================================================
# ASK LEGAL QUESTION
# ============================================================

if ask_button:

    # --------------------------------------------------------
    # FIRST CHECK CONNECTION
    # --------------------------------------------------------

    if not st.session_state.connected:

        st.error(
            "Please connect a valid OpenAI API key "
            "before asking a question."
        )

        st.stop()

    # --------------------------------------------------------
    # CHECK QUESTION
    # --------------------------------------------------------

    if not user_question.strip():

        st.warning(
            "Please enter a legal question."
        )

        st.stop()

    # --------------------------------------------------------
    # ADD USER MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        HumanMessage(
            content=user_question.strip()
        )
    )

    with st.chat_message("user"):

        st.write(
            user_question.strip()
        )

    # --------------------------------------------------------
    # CREATE CHAT MODEL
    # --------------------------------------------------------

    try:

        chat = ChatOpenAI(
            model=st.session_state.model,
            temperature=0.2,
            api_key=st.session_state.api_key,
        )

        # ----------------------------------------------------
        # SYSTEM MESSAGE
        # ----------------------------------------------------

        system_message = SystemMessage(
            content=(
                SYSTEM_PROMPT
                + "\n\n"
                + f"The selected LegalEase mode is: "
                f"{assistant_mode}"
            )
        )

        # ----------------------------------------------------
        # BUILD HISTORY
        # ----------------------------------------------------

        conversation = [
            system_message
        ] + st.session_state.messages

        # ----------------------------------------------------
        # AI RESPONSE
        # ----------------------------------------------------

        with st.chat_message("assistant"):

            with st.spinner(
                "Preparing a clear explanation..."
            ):

                response = chat.invoke(
                    conversation
                )

            st.write(
                response.content
            )

        # ----------------------------------------------------
        # SAVE RESPONSE
        # ----------------------------------------------------

        st.session_state.messages.append(
            AIMessage(
                content=response.content
            )
        )

    except AuthenticationError:

        st.error(
            "Your OpenAI API key is no longer valid. "
            "Please reconnect with a valid key."
        )

        st.session_state.connected = False

    except RateLimitError:

        st.error(
            "OpenAI returned a rate-limit or billing error. "
            "Please check your OpenAI account."
        )

    except APIConnectionError:

        st.error(
            "Could not connect to OpenAI. "
            "Please try again."
        )

    except APIError as error:

        st.error(
            f"OpenAI API error: {error}"
        )

    except Exception as error:

        st.error(
            f"Something went wrong: {error}"
        )


# ============================================================
# DISCLAIMER
# ============================================================

st.markdown(
    """
    <div class="disclaimer">
        LegalEase AI provides general legal information and does not
        replace advice from a qualified legal professional.
    </div>
    """,
    unsafe_allow_html=True,
)
