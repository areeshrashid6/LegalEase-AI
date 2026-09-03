import streamlit as st

from openai import (
    OpenAI,
    AuthenticationError,
    APIConnectionError,
    APIError,
    RateLimitError,
)

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
# LEGAL SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are LegalEase AI.

You are a DOMAIN-SPECIFIC legal information assistant.

Your ONLY purpose is to answer questions related to LAW and
LEGAL INFORMATION.

You are trained to help users with:

1. Legal terminology
2. Contract explanations
3. Legal document explanations
4. Basic rights information
5. Employment law concepts
6. Consumer rights concepts
7. Property and rental law concepts
8. Business and commercial law concepts
9. Family law concepts
10. Intellectual property concepts
11. Legal procedures
12. Court and dispute-resolution procedures
13. Questions to ask a lawyer
14. Basic legal document classification

============================================================
STRICT DOMAIN RULE
============================================================

ONLY answer questions that are substantially related to
legal information.

Examples of LEGAL questions:

- What does indemnification mean?
- What is a non-compete clause?
- Explain this contract clause.
- What are basic employee rights?
- What is a lease agreement?
- What is the difference between a plaintiff and defendant?
- What does breach of contract mean?
- How does a civil lawsuit generally work?
- What questions should I ask my lawyer?
- What type of legal document is this?

Examples of NON-LEGAL questions:

- How do I cook pasta?
- Write Python code.
- Who won a football match?
- What is the weather?
- Help me plan a vacation.
- Write a marketing email.
- Explain mathematics.
- Tell me a joke.
- What laptop should I buy?

For NON-LEGAL questions, DO NOT answer the question.

Instead respond exactly with:

"I don't have training for that topic. LegalEase AI is designed
specifically for legal-information questions. Please ask me
about legal terminology, contracts, rights, legal procedures,
or another legal topic."

============================================================
LEGAL SAFETY
============================================================

You provide GENERAL LEGAL INFORMATION, not legal advice.

Do not claim to be a lawyer.

Do not tell users that they definitely will win or lose a case.

Do not make definitive legal conclusions when important facts
are missing.

Laws vary by country, state, province and jurisdiction.

When jurisdiction matters, ask the user for their jurisdiction.

Never invent:

- laws
- statutes
- court cases
- legal citations
- deadlines
- penalties
- legal requirements

Explain complicated legal concepts in simple language.

Use headings and bullet points when useful.

Give examples when they make the concept easier to understand.

For important legal decisions, recommend consulting a qualified
lawyer.
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


/* ==========================================================
   SIDEBAR BRAND
   ========================================================== */

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


/* ==========================================================
   SIDEBAR SECTION
   ========================================================== */

.sidebar-label {
    color: #929aa6;

    font-size: 10px;
    font-weight: 700;

    letter-spacing: 1.2px;

    text-transform: uppercase;

    margin-top: 18px;
    margin-bottom: 8px;
}


/* ==========================================================
   CONNECTION STATUS
   ========================================================== */

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


/* ==========================================================
   ERROR
   ========================================================== */

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
   HERO
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
   QUESTION AREA
   ========================================================== */

.question-wrapper {
    max-width: 820px;

    margin: 20px auto 28px;
}

.question-label {
    color: #182230;

    font-size: 13px;

    font-weight: 700;

    margin-bottom: 8px;
}


/* ==========================================================
   TEXT AREA
   ========================================================== */

div[data-testid="stTextArea"] textarea {
    border: 1px solid #e0e4e9;

    border-radius: 13px;

    background: #ffffff;

    color: #182230;

    font-size: 14px;

    padding: 14px;

    min-height: 115px;
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
   FEATURE CARDS
   ========================================================== */

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
   INTRO CARD
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

    # --------------------------------------------------------
    # BRAND
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # AI CONNECTION
    # --------------------------------------------------------

    st.markdown(
        '<div class="sidebar-label">AI Connection</div>',
        unsafe_allow_html=True,
    )


    api_key_input = st.text_input(
        "OpenAI API Key",
        value=st.session_state.api_key,
        type="password",
        placeholder="sk-...",
    )


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
        "Your API key and selected model are checked when you connect."
    )


    # --------------------------------------------------------
    # CONNECT
    # --------------------------------------------------------

    if st.button(
        "Connect AI",
        type="primary",
        use_container_width=True,
    ):

        st.session_state.connection_error = ""

        st.session_state.connected = False


        if not api_key_input.strip():

            st.session_state.connection_error = (
                "Please enter your OpenAI API key."
            )

        else:

            try:

                # Create OpenAI client
                client = OpenAI(
                    api_key=api_key_input.strip()
                )


                # ------------------------------------------------
                # CHECK AUTHENTICATION
                # ------------------------------------------------

                client.models.list()


                # ------------------------------------------------
                # CHECK MODEL
                # ------------------------------------------------

                client.models.retrieve(
                    model_input
                )


                # ------------------------------------------------
                # SUCCESS
                # ------------------------------------------------

                st.session_state.api_key = (
                    api_key_input.strip()
                )

                st.session_state.model = model_input

                st.session_state.connected = True

                st.session_state.connection_error = ""


            except AuthenticationError:

                st.session_state.connection_error = (
                    "Invalid OpenAI API key. "
                    "Please check your key and try again."
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
    # ERROR MESSAGE
    # --------------------------------------------------------

    if st.session_state.connection_error:

        st.markdown(
            f"""
<div class="api-error">

    <strong>Connection failed</strong>
    <br><br>

    {st.session_state.connection_error}

</div>
""",
            unsafe_allow_html=True,
        )


    # --------------------------------------------------------
    # SUCCESS MESSAGE
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


    # --------------------------------------------------------
    # LEGAL ASSISTANT
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # CONVERSATION
    # --------------------------------------------------------

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

    st.caption("LegalEase AI")

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
# QUESTION AREA
# ============================================================

st.markdown(
"""
<div class="question-wrapper">

    <div class="question-label">
        Ask LegalEase
    </div>

</div>
""",
    unsafe_allow_html=True,
)


user_question = st.text_area(
    "Your legal question",
    placeholder=(
        "Example: What can my employer terminate me for "
        "without notice?"
    ),
    height=115,
    label_visibility="collapsed",
)


ask_col1, ask_col2, ask_col3 = st.columns(
    [1.2, 1, 1.2]
)


with ask_col2:

    ask_button = st.button(
        "⚖️  Ask LegalEase",
        type="primary",
        use_container_width=True,
    )


# ============================================================
# WELCOME FEATURES
# ============================================================

if not st.session_state.messages:

    st.markdown(
"""
<div class="intro-card">

    <div class="intro-title">
        Welcome to LegalEase
    </div>

    <div class="intro-text">
        Ask about a legal term, contract, right,
        procedure or legal document. LegalEase will
        explain it in simple language.
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


    for start in range(
        0,
        len(cards),
        3,
    ):

        row = cards[
            start:start + 3
        ]

        columns = st.columns(3)


        for column, card in zip(
            columns,
            row,
        ):

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
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    if isinstance(
        message,
        HumanMessage,
    ):

        with st.chat_message("user"):

            st.write(
                message.content
            )


    elif isinstance(
        message,
        AIMessage,
    ):

        with st.chat_message("assistant"):

            st.write(
                message.content
            )


# ============================================================
# LEGAL DOMAIN CLASSIFIER
# ============================================================

def check_if_legal_question(
    question,
    api_key,
    model,
):

    classifier_prompt = f"""
Determine whether the following user question is substantially
related to LAW or LEGAL INFORMATION.

Return ONLY one word:

LEGAL

or

NONLEGAL

Question:
{question}
"""


    classifier = ChatOpenAI(
        model=model,
        temperature=0,
        api_key=api_key,
    )


    result = classifier.invoke(
        [
            SystemMessage(
                content=(
                    "You are a strict legal-domain classifier. "
                    "Only classify questions that are genuinely "
                    "related to law or legal information as LEGAL."
                )
            ),
            HumanMessage(
                content=classifier_prompt
            ),
        ]
    )


    classification = (
        result.content
        .strip()
        .upper()
    )


    return classification == "LEGAL"


# ============================================================
# ASK LEGAL QUESTION
# ============================================================

if ask_button:


    # --------------------------------------------------------
    # CHECK CONNECTION
    # --------------------------------------------------------

    if not st.session_state.connected:

        st.error(
            "Please connect a valid OpenAI API key "
            "before asking a question."
        )

        st.stop()


    # --------------------------------------------------------
    # CHECK EMPTY QUESTION
    # --------------------------------------------------------

    if not user_question.strip():

        st.warning(
            "Please enter a legal question."
        )

        st.stop()


    # --------------------------------------------------------
    # CHECK LEGAL DOMAIN
    # --------------------------------------------------------

    try:

        is_legal = check_if_legal_question(
            user_question,
            st.session_state.api_key,
            st.session_state.model,
        )


    except AuthenticationError:

        st.error(
            "Your OpenAI API key is no longer valid. "
            "Please reconnect."
        )

        st.session_state.connected = False

        st.stop()


    except Exception as error:

        st.error(
            f"Unable to check the question: {error}"
        )

        st.stop()


    # --------------------------------------------------------
    # NON-LEGAL QUESTION
    # --------------------------------------------------------

    if not is_legal:

        with st.chat_message("user"):

            st.write(
                user_question
            )


        with st.chat_message("assistant"):

            st.warning(
                "I don't have training for that topic. "
                "LegalEase AI is designed specifically for "
                "legal-information questions. Please ask me "
                "about legal terminology, contracts, rights, "
                "legal procedures, or another legal topic."
            )


        st.stop()


    # --------------------------------------------------------
    # LEGAL QUESTION
    # --------------------------------------------------------

    st.session_state.messages.append(
        HumanMessage(
            content=user_question
        )
    )


    with st.chat_message("user"):

        st.write(
            user_question
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
                + f"The selected LegalEase focus is: "
                f"{assistant_mode}"
            )
        )


        # ----------------------------------------------------
        # CONVERSATION
        # ----------------------------------------------------

        conversation = (
            [system_message]
            + st.session_state.messages
        )


        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        with st.chat_message("assistant"):

            with st.spinner(
                "Preparing a clear legal explanation..."
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
