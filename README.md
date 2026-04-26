# QandA MCP Server

A Model Context Protocol (MCP) server that enables AI agents (Claude Desktop or any MCP-compatible client) to ask natural language questions over a corpus of PDF documents and receive grounded, source-attributed answers.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Setup Instructions](#setup-instructions)
3. [Tool Documentation](#tool-documentation)
4. [Example Interaction Log](#example-interaction-log)
5. [Vibe Coding Setup](#vibe-coding-setup)
6. [Known Limitations & Future Work](#known-limitations--future-work)

---

## Architecture Overview

```
documents/
  ├── NYSE_UNH_2020.pdf
  ├── FBS_PM_29NOV2023_Public.pdf
  ├── C18-1117.pdf
  ├── 2020.starsem-1.17.pdf
  └── 2020.findings-emnlp.139.pdf
        │
        ▼
  [Ingestion — src/ingestion/]
  PyPDFLoader → RecursiveCharacterTextSplitter
  (1000 chars, 200 overlap)
        │
        ▼
  [Retrieval — src/retrieval/]
  FAISS (dense, MMR, top-5)  +  BM25 (keyword, top-5)
  → deduplicated merge, capped at 8 docs
        │
        ▼
  [LangGraph Pipeline — src/graph/]
        │
        ├── retrieve       — hybrid FAISS + BM25 fetch
        ├── eval_each_doc  — LLM scores each chunk [0.0–1.0]
        │       │
        │    CORRECT (any score > 0.7)
        │       └──────────────────────────┐
        │    INCORRECT / AMBIGUOUS         │
        │       │                          │
        ├── rewrite_query                  │
        │   (LLM rewrites as web query)    │
        ├── web_search                     │
        │   (Tavily top-5 results)         │
        │                                  │
        └──────────────────────────────────┘
                          │
                       refine
                  (sentence-level LLM filter)
                          │
                       generate
                  (GPT-4o-mini answers from kept context)
                          │
                       answer + sources → MCP tool response
```

### Technology Stack

| Component | Technology | Details |
|---|---|---|
| **MCP Framework** | FastMCP | Handles MCP protocol compliance, tool registration, and server lifecycle via `@mcp.tool()` decorator |
| **PDF Ingestion** | LangChain PyPDFLoader | Loads each PDF page-by-page, preserving page number metadata used for source attribution |
| **Text Splitting** | RecursiveCharacterTextSplitter | Splits text into 1000-character chunks with 200-character overlap; no API calls at chunk time, keeping startup fast |
| **Embeddings** | OpenAI `text-embedding-3-large` | Used to build the FAISS vector store; high-dimensional embeddings for strong semantic search |
| **Dense Retrieval** | FAISS (MMR) | Retrieves top-5 semantically similar chunks using Maximal Marginal Relevance to reduce redundancy; candidate pool of 20 |
| **Keyword Retrieval** | BM25Retriever | Retrieves top-5 chunks based on exact keyword overlap; complements FAISS on acronym and term-specific queries |
| **Pipeline Orchestration** | LangGraph StateGraph | Manages the multi-step CRAG(Corrective RAG) flow as a directed graph with conditional routing between nodes |
| **LLM** | OpenAI `gpt-4o-mini` (temp=0.2) | Used for all reasoning tasks — chunk relevance grading, sentence filtering, query rewriting, and final answer generation |
| **Web Search** | TavilySearchResults | Fetches up to 5 live web results when PDF documents don't contain a confident answer |

### Module Structure

```
QandA_MCP_Server/
├── main.py                  # MCP server entry point, tool definitions
├── src/
│   ├── ingestion/           # PDF loading and text splitting
│   │   
│   ├── retrieval/           # FAISS + BM25 retriever construction
│   │  
│   ├── graph/               # LangGraph pipeline (nodes, routing, state)
│   │   
│   └── prompts/             # All LLM prompt templates
│       
├── documents/               # PDF corpus (not committed to git)
├── pyproject.toml
└── .env                     # API keys (not committed to git)
```

---

## Setup Instructions

### Prerequisites

- Python 3.10+
- [`uv`](https://github.com/astral-sh/uv) package manager
- OpenAI API key
- Tavily API key (free tier available at [tavily.com](https://tavily.com))

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd QandA_MCP_Server
```

### 2. Add your PDF documents

Place your PDF files in the `documents/` folder:

```
documents/
  ├── your_file_1.pdf
  └── your_file_2.pdf
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
```

### 4. Install dependencies

```bash
uv sync
```

### 5. Run the server

**Test via CLI:**
```bash
uv run python main.py
```

**Test via MCP Inspector:**
```bash
uv run fastmcp dev main.py
```

**Connect to Claude Desktop** — add this to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "qanda-mcp-server": {
      "command": "/path/to/uv",
      "args": [
        "run",
        "--project", "/absolute/path/to/QandA_MCP_Server",
        "python", "/absolute/path/to/QandA_MCP_Server/main.py"
      ]
    }
  }
}
```

On Windows the config file is at:
```
%APPDATA%\Claude\claude_desktop_config.json
```

> **Note:** On first query the server builds the FAISS index (OpenAI embedding API calls). This takes 20–60 seconds depending on corpus size. Subsequent queries are fast.

---

## Tool Documentation

This server exposes two MCP tools. The first is a simple utility tool and the second is the core of the system.

---

### Tool 1: `list_documents`

Lists all PDF files currently loaded in the knowledge base. Useful for knowing what documents the server has indexed before asking questions.

**Input:** None

**Output:** A numbered list of all indexed PDF filenames.

**Example output:**
```
Indexed documents (5):
1. 2020.findings-emnlp.139.pdf
2. 2020.starsem-1.17.pdf
3. C18-1117.pdf
4. FBS_PM_29NOV2023_Public.pdf
5. NYSE_UNH_2020.pdf
```

---

### Tool 2: `query_documents` — Corrective RAG (CRAG)

This is the core tool. Under the hood it implements **Corrective RAG (CRAG)**, which is an improvement over traditional RAG.

**What's wrong with traditional RAG?**

In a standard RAG pipeline, the user's question is converted into an embedding, the most semantically similar chunks are retrieved from the vector database, and those chunks are handed directly to the LLM as context. The problem is that this approach blindly trusts the retrieved chunks — it assumes they are all relevant and useful, which is often not true. Irrelevant or noisy chunks sent to the LLM can lead to hallucinated or incorrect answers.

**How CRAG fixes this**

CRAG adds an evaluation step after retrieval. Instead of passing chunks directly to the LLM for generation, it first asks the LLM to score each retrieved chunk for relevance to the question. Based on those scores, the system decides what to do next — rather than always generating from whatever was retrieved.

**Step-by-step pipeline for this tool:**

**1. Hybrid Retrieval**
The question is sent to both FAISS (dense semantic search, top-5 with MMR) and BM25 (keyword-based search, top-5). Results are merged and deduplicated, giving up to 8 candidate chunks. Using both retrievers improves recall — FAISS handles paraphrases and concepts, BM25 handles exact terms and acronyms.

**2. Evaluation (The CRAG step)**
Each of the retrieved chunks is individually scored by `gpt-4o-mini` on a scale of 0.0 to 1.0 for how useful it is in answering the question. Based on all scores, a verdict is assigned:
- **CORRECT** — at least one chunk scored above 0.7. The PDFs have a confident answer.
- **INCORRECT** — all chunks scored below 0.3. The PDFs have no useful information.
- **AMBIGUOUS** — scores fall in between. The PDFs have partial information.

**3. Routing**
- If **CORRECT** → proceed directly to refinement using the good chunks.
- If **INCORRECT** or **AMBIGUOUS** → the question is rewritten into a web search query by the LLM and sent to **Tavily**, which fetches up to 5 live web results. This is the fallback mechanism — instead of saying "I don't know" or hallucinating, the system goes to the web.

**4. Refinement**
Before generation, the context (from PDFs, web, or both) is broken down sentence by sentence. Each sentence is individually evaluated by the LLM — only sentences that directly help answer the question are kept. This strips out noise even from good documents.

**5. Generation**
The LLM generates the final answer using only the kept sentences as context. It is instructed to answer solely from the provided context and say "I don't know" if the context is insufficient.

**Input:**

| Parameter | Type | Description |
|---|---|---|
| `question` | `string` | A natural language question about the document content |

**Output:**

- The grounded answer
- `**Document Sources:**` — PDF filenames and page numbers (when PDFs had a confident answer)
- `**Web Sources:**` — title and URL of web results (when Tavily was used)
- A note indicating if a web search was triggered

**Example query:**
```
What are UnitedHealth Group's clinical programs for chronic conditions?
```

**Example output:**
```
**Document Sources:**
- NYSE_UNH_2020.pdf (Page 5)
- NYSE_UNH_2020.pdf (Page 12)

UnitedHealthcare Employer and Individual's clinical programs include condition-specific
programs for asthma, cancer, diabetes, heart disease, and kidney disease...
```

---

## Example Interaction Log

### Q1 — What is OptumRx?

**Question:** What is OptumRx?

**Response:**
```
**Document Sources:**
- NYSE_UNH_2020.pdf (Page 2)
- NYSE_UNH_2020.pdf (Page 5)
- NYSE_UNH_2020.pdf (Page 82)

OptumRx is a pharmacy care services provider that offers a full spectrum of services
through a network of over 67,000 retail pharmacies, home delivery, specialty and community
health pharmacies, and in-home and pharmacy infusion services. It manages limited and
ultra-limited distribution drugs in areas such as oncology, HIV, pain management, and
ophthalmology. OptumRx integrates pharmacy and medical care, utilizing demographic,
medical, laboratory, pharmaceutical, and clinical data to enhance clinical quality and
manage costs. It serves health benefits providers, employers, and government entities,
and offers clinical programs and digital tools to improve health outcomes and medication
adherence.
```

**Verdict:** CORRECT — answer drawn entirely from PDF documents.

---

### Q2 — Clinical programs for employer groups

**Question:** What does UnitedHealthcare Employer and Individual's clinical programs have?

**Response:**
```
**Document Sources:**
- NYSE_UNH_2020.pdf (Page 5)
- NYSE_UNH_2020.pdf (Page 9)

UnitedHealthcare Employer and Individual's clinical programs include condition-specific
programs designed to help employees manage complex or chronic conditions such as asthma,
cancer, diabetes, heart disease, and kidney disease. These programs provide personalized
care plans, timely outreach, clinical support, and personal guidance. They use data and
analytics to identify at-risk employees for targeted support across 50+ conditions.
Additional offerings include type 2 diabetes management solutions and Personal Health
Support, which screens employees for acute and chronic conditions to lower medical costs
and improve outcomes.
```

**Verdict:** CORRECT — answer drawn entirely from PDF documents.

---

### Q3 — Regulatory risks

**Question:** What are the Regulatory Trends and Uncertainties?

**Response:**
```
**Document Sources:**
- NYSE_UNH_2020.pdf (Page 14)
- NYSE_UNH_2020.pdf (Page 17)

UnitedHealth Group faces regulatory uncertainty from potential changes to the Affordable
Care Act, Medicare Advantage reimbursement rates, and Medicaid funding. The company is
subject to extensive federal and state regulation around health plan operations, pharmacy
benefit management, and data privacy. Key risks include shifts in government-sponsored
program funding, evolving price transparency requirements, and regulatory scrutiny of
vertical integration across insurance and care delivery businesses.
```

**Verdict:** CORRECT — answer drawn entirely from PDF documents.

---

## Vibe Coding Setup

### Tool Used

**Claude Code** 

### How I Used It

Claude Code was used throughout the project as a development aid — helping implement the pipeline, refactor the codebase into a clean modular structure, and debug integration issues. Some specific areas where it added real value:

- **Pipeline optimization** — Used Claude to sanity-check design decisions and suggest improvements during development, such as retrieval configuration, prompt quality for the evaluator and filter chains, and how context is prepared before being sent to the LLM.
- **Refactoring** — Once the core pipeline was working, Claude assisted in breaking it out from a single file into a clean `src/<module>/__init__.py` package structure without breaking existing behavior.
- **Debugging** — Useful for diagnosing multi-layer integration errors, particularly around MCP server connectivity and path resolution issues when launched from Claude Desktop.

> **Note:** Claude Code requires an Anthropic subscription. A solid free alternative is **[Open Code](https://open-vsx.org/extension/opencode-ai/opencode)** (VS Code extension), which gives access to several free-tier models including Nvidia's **Nemotron** — capable models that work well for most coding tasks without any cost.

---

## Known Limitations & Future Work

### Semantic Chunking

The biggest area for improvement in this pipeline is how documents are chunked. Chunking is foundational to any RAG system — the quality of chunks directly affects what gets retrieved, and what gets retrieved directly affects what the LLM can answer from.

Currently the server uses `RecursiveCharacterTextSplitter` from LangChain, which splits text into chunks of 1000 characters with a 200-character overlap, respecting natural separators like paragraphs and sentences. It is a solid, well-tested default that works reliably across document types. That said, there is room to push chunk quality further.

The better alternative is LangChain's `SemanticChunker` (available under `langchain_experimental`). Instead of splitting by character count, it works by:

1. Breaking the document into individual sentences
2. Computing an embedding for each sentence using the configured embedding model
3. Measuring the cosine distance between each pair of consecutive sentence embeddings
4. Identifying breakpoints where the semantic distance jumps significantly (using a percentile threshold) — indicating a topic shift
5. Grouping sentences between breakpoints into a single chunk

The result is chunks that are semantically coherent — each chunk covers one idea or topic rather than being an arbitrary slice of text. This meaningfully improves retrieval quality because the vector store is indexing units of meaning rather than units of character length.

The reason it was not used here is a latency trade-off. Because `SemanticChunker` calls the embedding API for every sentence in every document at startup, the indexing step is significantly slower — slow enough that the MCP client (Claude Desktop) would time out waiting for the server to become ready before it had finished building the index. Additionally, the splitter is still housed in `langchain_experimental`, signalling it is not yet considered stable for production use.

The trade-off between chunk quality and startup latency was not compelling enough to justify it for this implementation, but switching to `SemanticChunker` would be the most impactful improvement to make next.
