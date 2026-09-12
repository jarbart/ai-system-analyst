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

    response = requests.post(
        "http://127.0.0.1:8000/analyze",
        json={"query": query},
        timeout=120,
    )

    response.raise_for_status()

    data = response.json()

    st.subheader("Analysis")
    st.write(data["analysis"])

    st.subheader("Evidence")

    for evidence in data["evidence"]:
        with st.expander(
            f"{evidence['source_type']} · "
            f"{evidence['document_id']} · "
            f"{evidence['chunk_id']}"
        ):
            st.write(f"Score: {evidence['score']:.4f}")
            st.write(evidence["content"])