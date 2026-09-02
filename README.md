# LegalEase AI

A Streamlit Community Cloud app for explaining legal concepts in simple language.

## Features

- Explain legal terminology
- Explain contracts
- Summarize legal documents (prompt-based in this starter version)
- Explain basic rights
- Generate questions for a lawyer
- Explain legal procedures
- Legal document classification (prompt-based in this starter version)

## Run locally

1. Create a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start Streamlit:

```bash
streamlit run app.py
```

## Streamlit Community Cloud

Upload this folder to a GitHub repository, then deploy the repository from
Streamlit Community Cloud.

This version lets the user enter an OpenAI API key in the sidebar. The key is
kept in Streamlit session state for the current session and is not written to
a project file.

For a production app, prefer a server-side secret/backend rather than asking
every user for an API key.

## Important

LegalEase AI provides general legal information and is not a substitute for
advice from a qualified lawyer. Answers can be incomplete or incorrect, so
users should verify important legal matters with an appropriate professional.
