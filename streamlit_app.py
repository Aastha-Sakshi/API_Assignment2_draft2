"""
Interactive demo UI — AI Recruitment Assistant.

    Terminal 1:  uvicorn app.main:app --reload --port 8000
    Terminal 2:  streamlit run streamlit_app.py

The UI holds no model code: every tab is an HTTP call to the FastAPI service.
That is deliberate — the assignment is about an API-driven application, and
this keeps the demo honest about where the work happens.
"""

import json
import os

import requests
import streamlit as st

API = os.getenv("API_BASE", "http://127.0.0.1:8000")
TIMEOUT = 180

st.set_page_config(page_title="AI Recruitment Assistant", layout="wide", page_icon="🧭")


def call(method: str, path: str, **kwargs):
    try:
        response = requests.request(method, f"{API}{path}", timeout=TIMEOUT, **kwargs)
    except requests.RequestException as exc:
        st.error(f"Cannot reach the API at {API} — is uvicorn running?\n\n{exc}")
        return None
    if not response.ok:
        st.error(f"{response.status_code}: {response.json().get('detail', response.text)}")
        return None
    return response.json()


def show_model(block: dict | None):
    if not block:
        return
    model = block.get("model") or {}
    bits = [f"**model** `{model.get('model', '?')}`"]
    if model.get("category"):
        bits.append(f"**category** `{model['category']}`")
    if model.get("prompt_version"):
        bits.append(f"**prompt** `{model['prompt_version']}`")
    if block.get("latency_sec") is not None:
        bits.append(f"**latency** `{block['latency_sec']}s`")
    if block.get("total_tokens"):
        bits.append(f"**tokens** `{block['total_tokens']}`")
    st.caption(" · ".join(bits))
    if block.get("degraded"):
        st.warning(f"Degraded to local fallback — {block.get('degraded_reason', 'LLM unavailable')}")


# ---------------------------------------------------------------- sidebar --
with st.sidebar:
    st.title("🧭 Recruitment Assistant")
    health = call("GET", "/health")
    if health:
        st.success("API online")
        st.metric("LLM backend", health["llm_backend"])
        st.write("Fine-tuned model:", "✅ loaded" if health["finetuned_model_loaded"] else "❌ not trained yet")
        st.caption(f"v{health['app_version']}")
    st.divider()
    st.caption("CCZG506 Assignment II · HR/Recruitment domain")
    st.caption("**CV:** doc classification, OCR  \n**NLP:** NER, fit classification, QA, generation")

# --------------------------------------------------------------- session ---
st.session_state.setdefault("resume_text", "")
st.session_state.setdefault("job_description", "")

tabs = st.tabs([
    "① Upload (CV)",
    "② Entities (NLP)",
    "③ Fit (fine-tuned)",
    "④ Ask (NLP)",
    "⑤ Brief (NLP)",
    "⑥ Full screening",
    "📊 LLMOps",
])

# ------------------------------------------------- 1: CV sub-tasks 1 & 2 ---
with tabs[0]:
    st.header("Sub-tasks 1 & 2 — Computer Vision")
    st.write("DiT verifies the upload is a resume, then the text layer or docTR OCR reads it.")

    upload = st.file_uploader("Resume (PDF / PNG / JPG / TXT)", type=["pdf", "png", "jpg", "jpeg", "txt"])
    if upload and st.button("Ingest document", type="primary"):
        with st.spinner("Classifying document and extracting text..."):
            result = call("POST", "/ingest-resume", files={"file": (upload.name, upload.getvalue())})
        if result:
            if result["warning"]:
                st.warning(result["warning"])
            left, right = st.columns(2)
            with left:
                doc_type = result["document_type"]
                if doc_type:
                    st.subheader("① Document type")
                    st.metric(doc_type["predicted_type"], f"{doc_type['confidence']:.1%}")
                    st.bar_chart({p["label"]: p["score"] for p in doc_type["top_k"]})
                    show_model(doc_type)
                else:
                    st.info("Plain-text upload — image classification skipped.")
            with right:
                st.subheader("② Extracted text")
                extraction = result["extraction"]
                st.caption(f"method `{extraction['extraction_method']}` · {extraction['char_count']} chars")
                st.text_area("Text", extraction["text"], height=300, key="extracted_preview")
            st.session_state.resume_text = result["extraction"]["text"]
            st.success("Resume text stored — the other tabs now use it.")

    st.divider()
    st.subheader("Or paste directly")
    pasted = st.text_area("Resume text", st.session_state.resume_text, height=200)
    if pasted != st.session_state.resume_text:
        st.session_state.resume_text = pasted

    st.session_state.job_description = st.text_area(
        "Job description", st.session_state.job_description, height=180,
        placeholder="Paste the JD here — used by tabs ③, ⑤ and ⑥",
    )

# ------------------------------------------------------------- 2: NER -----
with tabs[1]:
    st.header("Sub-task 3 — Named Entity Recognition")
    if st.button("Extract entities", disabled=not st.session_state.resume_text):
        with st.spinner("Running NER..."):
            result = call("POST", "/entities", json={"text": st.session_state.resume_text})
        if result:
            st.caption(f"{result['count']} unique entities across {result['chunks_processed']} chunk(s)")
            for group, values in result["grouped"].items():
                st.write(f"**{group}** — {', '.join(values)}")
            st.dataframe(result["entities"], use_container_width=True)
            show_model(result)

# -------------------------------------------------- 3: fit classification -
with tabs[2]:
    st.header("Sub-task 4 — Fit classification (the fine-tuned model)")
    st.write("Fine-tuned DistilBERT vs the same task handed to gpt-oss-20b as a prompt.")

    ready = bool(st.session_state.resume_text and st.session_state.job_description)
    if not ready:
        st.info("Provide both a resume and a job description in tab ①.")

    col1, col2 = st.columns(2)
    payload = {
        "resume_text": st.session_state.resume_text,
        "job_description": st.session_state.job_description,
    }

    with col1:
        if st.button("Run fine-tuned model", disabled=not ready, type="primary"):
            result = call("POST", "/classify-fit", json={**payload, "method": "finetuned"})
            if result:
                st.metric("Fine-tuned verdict", result["label"], f"{result['confidence']:.1%} confidence")
                st.bar_chart(result["scores"])
                show_model(result)

    with col2:
        if st.button("Run prompted LLM", disabled=not ready):
            result = call("POST", "/classify-fit", json={**payload, "method": "prompted"})
            if result:
                st.metric("Prompted verdict", result["label"] or "unparsable")
                st.code(result["raw_output"] or "(empty)")
                show_model(result)

    if st.button("Compare both side by side", disabled=not ready):
        result = call("POST", "/compare-fit-models", json=payload)
        if result:
            st.write("**Agreement:**", "✅ same verdict" if result["agree"] else "❌ they disagree")
            st.json(result)

# ------------------------------------------------------------- 4: QA ------
with tabs[3]:
    st.header("Sub-task 5 — Extractive Question Answering")
    st.caption("Extractive, so every answer is a literal span of the resume — it cannot hallucinate a qualification.")
    question = st.text_input("Question", "How many years of experience does the candidate have?")
    if st.button("Ask", disabled=not st.session_state.resume_text):
        result = call("POST", "/ask", json={"resume_text": st.session_state.resume_text, "question": question})
        if result:
            # On abstention the model found no span above threshold, so
            # "confidence" is confidence that the question is unanswerable —
            # showing it as answer confidence next to "Not stated" reads as the
            # opposite of what happened.
            if result["grounded"]:
                st.success(result["answer"])
                st.caption(
                    f"confidence {result['confidence']:.1%} · "
                    f"resume chars {result['start_char']}–{result['end_char']}"
                )
            else:
                st.info(result["answer"])
                st.caption(
                    f"No span in the resume answered this. "
                    f"Unanswerable score {result['abstention_score']:.1%}."
                )
            show_model(result)

# ---------------------------------------------------------- 5: generation -
with tabs[4]:
    st.header("Sub-task 6 — Candidate brief + interview questions")
    if st.button("Generate brief", disabled=not st.session_state.resume_text, type="primary"):
        with st.spinner("Calling gpt-oss-20b..."):
            result = call("POST", "/candidate-brief", json={
                "resume_text": st.session_state.resume_text,
                "job_description": st.session_state.job_description,
            })
        if result:
            st.markdown(result["brief"])
            show_model(result)

# ------------------------------------------------------- 6: orchestration -
with tabs[5]:
    st.header("Full screening — all sub-tasks, one call")
    st.caption("`POST /screen-candidate` chains NER → fit classification → brief. This is the unified objective.")
    method = st.radio("Fit model", ["auto", "finetuned", "prompted"], horizontal=True)
    if st.button("Screen candidate", type="primary",
                 disabled=not (st.session_state.resume_text and st.session_state.job_description)):
        with st.spinner("Running the full pipeline..."):
            result = call("POST", "/screen-candidate", json={
                "resume_text": st.session_state.resume_text,
                "job_description": st.session_state.job_description,
                "method": method,
            })
        if result:
            st.metric("Verdict", result["fit"]["label"])
            st.caption(f"total pipeline latency {result['pipeline_latency_sec']}s")
            if result["brief"]:
                st.markdown(result["brief"]["brief"])
            st.info(result["disclaimer"])
            with st.expander("Raw JSON response"):
                st.json(result)

# ---------------------------------------------------------------- 7: ops --
with tabs[6]:
    st.header("LLMOps metrics")
    if st.button("Refresh /metrics", type="primary"):
        st.session_state["metrics"] = call("GET", "/metrics")

    data = st.session_state.get("metrics")
    if data and "session" in data:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Requests", data["session"]["total_requests"])
        c2.metric("p95 latency", f"{data['M1_latency_sec']['overall']['p95']}s")
        c3.metric("Success rate", f"{data['M3_reliability']['success_rate']:.1%}")
        c4.metric("Total tokens", data["M4_token_usage"]["total_tokens"])

        st.subheader("M1 · Latency by endpoint")
        st.bar_chart({ep: stats["p95"] for ep, stats in data["M1_latency_sec"]["by_endpoint"].items()})

        st.subheader("M2 · Request volume by endpoint")
        st.bar_chart(data["M2_throughput"]["requests_by_endpoint"])

        st.subheader("M7 · Fine-tuned vs prompted (offline)")
        st.json(data["M7_quality_offline"])

        with st.expander("Full metrics payload"):
            st.code(json.dumps(data, indent=2), language="json")
    elif data:
        st.info(data.get("message", "No data yet."))
