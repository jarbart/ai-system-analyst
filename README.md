# AI System Analyst

AI-powered incident investigation assistant for system analysts and engineers.

The project combines document ingestion, hybrid retrieval, reranking, local embeddings and a local LLM to investigate production incidents and present the analysis together with the retrieved evidence.

The application is designed as a small, resource-efficient AI Engineering project that can run locally without cloud APIs or GPU acceleration.

---

## Project goal

The goal of the project is to explore how an AI assistant could support system analysis and incident investigation.

Given an incident description such as:

> Why does the Orders API return HTTP 500?

the system:

1. searches the available knowledge base,
2. combines semantic and keyword retrieval,
3. reranks the retrieved chunks,
4. assembles an evidence-based context,
5. sends the context to a local LLM,
6. validates the LLM response against a structured schema,
7. presents the result through a REST API and Streamlit UI.

The assistant is designed to distinguish between confirmed evidence, hypotheses and recommended next steps.

---

## Architecture

```text
                 Knowledge Base
          ┌──────────────────────────┐
          │ Documentation             │
          │ Incidents                 │
          │ Logs / Requirements       │
          └────────────┬─────────────┘
                       │
                       ▼
                   Ingestion
                       │
                       ▼
               Chunking + Metadata
                       │
                       ▼
              ┌───────────────────┐
              │ Retrieval Layer   │
              │                   │
              │ Vector Search     │
              │ Keyword Search    │
              └─────────┬─────────┘
                        │
                        ▼
                  Hybrid Search
                        │
                        ▼
                    Reranking
                        │
                        ▼
                Context Assembly
                        │
                        ▼
                    Local LLM
                Ollama / Qwen 0.5B
                        │
                        ▼
              Structured Analysis
                        │
                        ▼
                     FastAPI
                        │
                        ▼
                    Streamlit
```

---

## Main components

### Domain model

The project defines explicit domain objects for:

* documents,
* chunks,
* source types,
* incident analysis results,
* evidence.

This keeps the core application logic independent from infrastructure details.

### Ingestion

The current implementation loads text-based knowledge sources and converts them into domain `Document` objects.

Documents are split into overlapping chunks with metadata such as:

* document ID,
* source type,
* chunk ID,
* chunk position.

### Hybrid retrieval

The retrieval layer combines two approaches:

**Semantic search**

Uses local sentence-transformer embeddings and FAISS vector search.

**Keyword search**

Uses TF-IDF-based lexical retrieval.

The two result sets are combined using Reciprocal Rank Fusion (RRF).

This provides a simple hybrid retrieval strategy that works well for a small local knowledge base.

### Reranking

Retrieved candidates are passed through a lightweight reranking layer before context construction.

The reranker is intentionally simple because the project is designed to run on limited hardware.

### Context assembly

The retrieved chunks are converted into a structured context containing source metadata.

Example:

```text
[SOURCE: incident]
[DOCUMENT: incident-1001]
[CHUNK: incident-1001-0001]

ERROR OrdersRepository - database connection timeout after 5 seconds
...
```

This gives the LLM explicit evidence identifiers and source information.

### Local LLM

The project uses:

* Ollama
* Qwen 2.5 0.5B

The model runs locally on CPU.

The LLM is instructed to return a structured incident analysis containing:

* `root_cause`
* `confirmed_evidence`
* `hypotheses`
* `next_steps`

The response is validated with Pydantic.

Ollama is configured with the JSON schema generated from the Pydantic model rather than relying only on a generic JSON response.

This is important because small local models can otherwise return structurally invalid JSON.

### API

FastAPI exposes:

```text
POST /analyze
```

Example request:

```json
{
  "query": "Why does the Orders API return HTTP 500?"
}
```

The response contains the structured analysis together with the retrieved evidence.

### UI

The Streamlit interface provides a simple incident investigation workflow.

The user enters an incident description and receives:

* root cause,
* confirmed evidence,
* hypotheses,
* next steps,
* retrieved evidence.

---

## Example scenario

The example knowledge base contains an incident involving the Orders API.

Relevant observations include:

* HTTP 500 responses from `GET /api/orders/{order_id}`,
* PostgreSQL connection timeout errors,
* elevated database CPU usage,
* active connections close to the configured connection limit,
* increased traffic to the Orders service.

The assistant can use these sources to produce an analysis such as:

```text
Root cause:
Orders API requests are failing because database connections are timing out.

Hypothesis:
The Orders service may be exhausting the available PostgreSQL connections
under increased traffic.

Next steps:
- Check the PostgreSQL connection limit.
- Check the number of active connections.
- Review connection pool configuration.
- Check whether connections are being released correctly.
```

The retrieved chunks are displayed alongside the analysis so that the user can inspect the evidence used by the system.

---

## Technology stack

### Application

* Python 3.12
* FastAPI
* Streamlit
* Pydantic

### AI / RAG

* Ollama
* Qwen 2.5 0.5B
* sentence-transformers
* FAISS
* scikit-learn

### Quality

* pytest
* Ruff
* mypy

### Development environment

* Ubuntu
* Python virtual environment
* local CPU inference

The application does not require a GPU.

---

## Project structure

```text
ai-system-analyst/
├── data/
│   └── knowledge/
│       ├── incident-1001.txt
│       └── orders-api.txt
│
├── src/
│   └── ai_system_analyst/
│       ├── api/
│       ├── domain/
│       ├── ingestion/
│       ├── llm/
│       ├── retrieval/
│       ├── services/
│       └── ui/
│
├── tests/
│
├── pyproject.toml
├── README.md
└── .gitignore
```

---

## Running locally

### 1. Create and activate the virtual environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 2. Install the project

```bash
pip install -e .
```

For development tools:

```bash
pip install -e ".[dev]"
```

### 3. Install Ollama

Install Ollama using the official installation method for your operating system.

Then download the model:

```bash
ollama pull qwen2.5:0.5b
```

Make sure Ollama is running and available on:

```text
http://127.0.0.1:11434
```

### 4. Start FastAPI

From the project root:

```bash
source .venv/bin/activate

uvicorn ai_system_analyst.api.app:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

### 5. Start Streamlit

In another terminal:

```bash
source .venv/bin/activate

streamlit run src/ai_system_analyst/ui/streamlit_app.py
```

Open the displayed Streamlit URL in a browser.

---

## Testing

Run the complete test suite:

```bash
pytest -q
```

The project also uses Ruff:

```bash
ruff check .
```

and mypy:

```bash
mypy src
```

The integration test for Ollama requires the local model to be available.

---

## Design decisions

### Local-first architecture

The application intentionally uses local inference instead of a hosted LLM API.

This makes the project:

* inexpensive to run,
* independent of cloud API keys,
* suitable for experimentation,
* easier to reproduce locally.

The trade-off is lower model quality and higher latency on CPU.

### Small model

Qwen 2.5 0.5B was selected because the project is intended to run on limited hardware.

This is not intended to represent the best possible LLM quality. Instead, it demonstrates that the surrounding AI Engineering architecture can be built independently of a large proprietary model.

### Hybrid retrieval

Pure semantic retrieval can miss exact technical terms such as:

* endpoint names,
* error messages,
* configuration keys,
* identifiers.

Keyword retrieval helps preserve these exact matches.

Combining both approaches provides a more practical retrieval strategy for technical documentation.

### Structured LLM output

The LLM output is represented by a Pydantic model.

The same model is used to generate a JSON schema supplied to Ollama.

This creates an explicit contract between the application and the LLM instead of treating the model response as arbitrary text.

---

## Current limitations

This project is intentionally a small prototype rather than a production incident-management platform.

Current limitations include:

* the knowledge base is loaded from local text files,
* FAISS storage is in memory,
* the index is rebuilt when the application starts,
* the keyword index is simple and not optimized for large datasets,
* reranking is lightweight,
* there is no persistent database,
* there is no authentication or authorization,
* the Streamlit API URL is currently configured directly in the application,
* the local 0.5B model has limited reasoning and language capabilities,
* LLM-generated conclusions still require human verification.

The system should therefore be treated as an **analyst assistance tool**, not an autonomous incident-response system.

---

## What this project demonstrates

The project demonstrates practical experience with:

* Python application architecture,
* domain modelling,
* REST API development,
* FastAPI dependency management,
* RAG architecture,
* document chunking,
* vector search,
* keyword search,
* hybrid retrieval,
* reranking,
* context construction,
* local LLM integration,
* structured LLM outputs,
* Pydantic validation,
* automated testing,
* static type checking,
* linting,
* local AI development on constrained hardware,
* end-to-end integration testing.

The project also demonstrates an important AI Engineering principle:

> The LLM is only one component of the system.

The retrieval, evidence handling, validation and application architecture are equally important parts of building a useful AI system.

---

## Status

**Mini-project completed.**

The current implementation provides a working end-to-end incident analysis flow:

```text
Knowledge Base
      ↓
Retrieval
      ↓
Reranking
      ↓
Context
      ↓
Local LLM
      ↓
Structured Analysis
      ↓
FastAPI
      ↓
Streamlit
```

Future improvements are possible, but the current version intentionally stops here as a complete portfolio mini-project.
