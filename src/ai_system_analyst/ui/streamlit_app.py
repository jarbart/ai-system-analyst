import requests
import streamlit as st

st.set_page_config(
    page_title="AI System Analyst",
    page_icon="🔎",
)

st.title("AI System Analyst")
st.write("Incident investigation assistant")

query = st.text_area(
    "Describe the incident",
    placeholder="Why does the Orders API return HTTP 500?",
)

if st.button("Analyze"):
    if not query.strip():
        st.warning("Please describe the incident before running the analysis.")
        st.stop()

    with st.spinner("Analyzing incident..."):
        response = requests.post(
            "http://127.0.0.1:8000/analyze",
            json={"query": query},
            timeout=120,
        )

    response.raise_for_status()

    data = response.json()

    st.subheader("Root cause")
    st.write(data["root_cause"])

    st.subheader("Confirmed evidence")

    if data["confirmed_evidence"]:
        for item in data["confirmed_evidence"]:
            st.write(f"- {item}")
    else:
        st.info("No confirmed evidence was identified.")

    st.subheader("Hypotheses")

    if data["hypotheses"]:
        for item in data["hypotheses"]:
            st.write(f"- {item}")
    else:
        st.info("No hypotheses were identified.")

    st.subheader("Next steps")

    if data["next_steps"]:
        for item in data["next_steps"]:
            st.write(f"- {item}")
    else:
        st.info("No next steps were identified.")

    st.subheader("Retrieved evidence")

    for evidence in data["evidence"]:
        with st.expander(
            f"{evidence['source_type']} · "
            f"{evidence['document_id']} · "
            f"{evidence['chunk_id']}"
        ):
            st.write(f"Score: {evidence['score']:.4f}")
            st.write(evidence["content"])