# QandA MCP Server

A Model Context Protocol (MCP) server that enables AI agents (Claude Desktop or any MCP-compatible client) to ask natural language questions over a corpus of PDF documents and receive grounded, source-attributed answers.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Setup Instructions](#setup-instructions)
3. [Tool Documentation](#tool-documentation)
4. [Example Interaction Log](#example-interaction-log)
5. [Vibe Coding Setup](#vibe-coding-setup)
6. [Limitations, Reflections & Future Work](#limitations-reflections--future-work)

---

## Architecture Overview

```
documents/
  ├── file1.pdf
  └── file2.pdf
        │
        ▼
  [Ingestion — src/ingestion/]
  PyPDFLoader → RecursiveCharacterTextSplitter
  (1000 chars, 200 overlap)
        │
        ▼
  [Retrieval — src/retrieval/]
  FAISS (dense, MMR, top-5) + BM25 (keyword, top-5)
  → deduplicated, capped at 8 docs
        │
        ▼
  [LangGraph Pipeline — src/graph/]
        │
        ├── retrieve       — hybrid FAISS + BM25 fetch
        │
        ├── eval_each_doc  — LLM scores each chunk [0.0–1.0]
        │
        │   ┌───────────┐  ┌───────────┐  ┌───────────┐
        │   │ INCORRECT │  │ AMBIGUOUS │  │  CORRECT  │
        │   │ all < 0.3 │  │  (mixed)  │  │ any > 0.7 │
        │   │  web only │  │ PDF + web │  │  PDF only │
        │   └─────┬─────┘  └─────┬─────┘  └─────┬─────┘
        │         └──────┬───────┘              │
        │                │                      │
        ├── rewrite_query + web_search (Tavily) | 
        │                └──────────────────────┘
        │                          │
        ├── refine ------> sentence-level LLM filter
        │                          |
        └── generate-----> GPT-4o-mini (temp=0.2)
                  │
                  ▼
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
├── documents/               # PDF corpus 
├── pyproject.toml
└── .env                     # API keys 
```

---

## Setup Instructions

### Prerequisites

- Python 3.10+
- OpenAI API key
- Tavily API key (free tier available at [tavily.com](https://tavily.com))

### 1. Install uv

```bash
pip install uv
```

> **Note:** On some Windows setups `uv` is not recognised as a command after installation. Use `python -m uv` as a prefix for all `uv` commands instead (e.g. `python -m uv sync`).

### 2. Clone the repository

```bash
git clone https://github.com/vatsaly-thakur-1610/QandA_MCP_Server.git
cd QandA_MCP_Server
```

### 3. Add your PDF documents

The `documents/` folder already exists. Drop your PDF files into it:

```
documents/
  ├── your_file_1.pdf
  └── your_file_2.pdf
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
```

### 5. Install dependencies

```bash
python -m uv sync
```

> **Note:** On first run the server builds the FAISS index (OpenAI embedding API calls). This takes a few seconds depending on corpus size. Subsequent runs are fast.

### 6. Connect to Claude Desktop

Find your `uv` executable path:

```bash
pip show uv
```

Take the `Location` from the output — `uv.exe` is in the `Scripts` folder one level up from it.

Add this to your `claude_desktop_config.json` (Which you can find by opening claude desktop -> click on your profile -> go to settings -> select developers tab -> click on Edit Config or an alternate way can be found on Windows at `%APPDATA%\Claude\claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "qanda-mcp-server": {
      "command": "C:\\path\\to\\Scripts\\uv.exe",
      "args": [
        "run",
        "--project", "C:\\absolute\\path\\to\\QandA_MCP_Server",
        "python", "C:\\absolute\\path\\to\\QandA_MCP_Server\\main.py"
      ]
    }
  }
}
```

Restart Claude Desktop after saving. Again go to the developers section to see wether the mcp server is up and running you should see something like this qanda-mcp-server - Name of our MCP server and the status which shows "running"
<img width="1712" height="822" alt="Screenshot 2026-04-26 103025" src="https://github.com/user-attachments/assets/ba5547d2-d3a7-4acd-b635-3835e8c0fe39" />


---

## Tool Documentation

This server exposes two MCP tools. The first is a simple utility tool and the second is the core of the system.

---

### Tool 1: `list_documents`

Lists all PDF files currently loaded in the knowledge base. Useful for knowing what documents the server has indexed before asking questions.

**Input:** Ask for the Documents in knowledge Base

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

This is the core tool. Under the hood it implements **Corrective RAG (CRAG)**, which is a significant improvement over traditional RAG.

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
- If **CORRECT** → proceed directly to refinement using the good chunks from the PDFs.
- If **INCORRECT** → all chunks scored below 0.3, meaning the PDFs have no useful information. The question is rewritten as a web search query and sent to **Tavily** (top-5 results). Only web results are used in refinement.
- If **AMBIGUOUS** → scores fall in between. The question is also sent to Tavily, but both the good PDF chunks and the web results are merged and used together in refinement.

**4. Refinement**
Before generation, the context (from PDFs, web, or both) is broken down sentence by sentence. Each sentence is individually evaluated by the LLM — only sentences that directly help answer the question are kept. This strips out noise even from good documents.

**5. Generation**
The LLM generates the final answer using only the kept sentences as context. It is instructed to answer solely from the provided context and say "I don't know" if the context is insufficient.

<img width="195" height="688" alt="Screenshot 2026-04-26 105032" src="https://github.com/user-attachments/assets/1de23472-67af-4e24-b2eb-4aad55bb5809" />

**Input:**

| Parameter | Type | Description |
|---|---|---|
| `question` | `string` | A natural language question about the document content |

**Output:**

The grounded answer along with sources cited — document names and page numbers from PDFs, or web source titles and URLs if a web search was triggered.

---

## Example Interaction Log

### Interaction 1 — Listing the knowledge base (`list_documents`)

**Prompt:** List all the documents in the knowledge base.

**Response:**
```
Indexed documents (6):
1. 2020.findings-emnlp.139.pdf
2. 2020.starsem-1.17.pdf
3. C18-1117.pdf
4. FBS_PM_29NOV2023_Public.pdf
5. NYSE_UNH_2020.pdf
6. Nexla_Software Engineer_Assignment.pdf
```
<img width="1037" height="770" alt="Screenshot 2026-04-26 103025" src="https://github.com/user-attachments/assets/e900c568-8731-4a61-ac2a-1441aadc1a77" />
<img width="843" height="565" alt="Screenshot 2026-04-26 103658" src="https://github.com/user-attachments/assets/7048bf74-461e-4bf1-8f36-477922574a19" />
---

### Interaction 2 — Querying across documents (`query_documents`)

**Question:** What is DST?

**Response:**
```
DST stands for Dialog State Tracking. It involves maintaining a set of
⟨domain, slot, value⟩ triplets to track all the states accumulated
throughout the conversational turns in a dialogue system.

--- SOURCES CITED ---
Document Sources:
- 2020.starsem-1.17.pdf (Page 0)
```
<img width="932" height="531" alt="Screenshot 2026-04-26 103942" src="https://github.com/user-attachments/assets/480d3a09-7bb7-48a8-8591-9be86c4910a7" />

---

### Interaction 3 — Querying a specific document (`query_documents`)

**Question:** What does UnitedHealthcare Employer & Individual's clinical programs include?

**Response:**
```
UnitedHealthcare Employer & Individual's clinical programs include:
- Wellness programs
- Decision support
- Utilization management
- Case and disease management
- Complex condition management
- On-site programs (such as biometrics and flu shots)
- Incentives for positive behavior change
- Mental health / substance use disorder management
- Employee assistance programs

--- SOURCES CITED ---
Document Sources:
- NYSE_UNH_2020.pdf (Page 7)

```
<img width="952" height="652" alt="Screenshot 2026-04-26 104028" src="https://github.com/user-attachments/assets/b12943db-0202-4281-a7d9-7593c3523a64" />

---

## Vibe Coding Setup

### Tool Used

**Claude Code** 

### How I Used It

Claude Code was used throughout the project as a development aid — helping implement the pipeline, refactor the codebase into a clean modular structure, and debug integration issues. Some specific areas where it added real value:

- **Pipeline optimization** — Used Claude to sanity-check design decisions and suggest improvements during development, such as retrieval configuration, prompt quality for the evaluator and filter chains, and how context is prepared before being sent to the LLM.
- **Refactoring** — Once the core pipeline was working, Claude assisted in breaking it out from a single file into a clean modular package structure without breaking existing behavior.
- **Debugging** — Useful for diagnosing multi-layer integration errors, particularly around MCP server connectivity and path resolution issues when launched from Claude Desktop.
- **Code review** — Used Claude Code to review the final codebase for quality, consistency, and any issues that are easy to miss when deep in the implementation.

**Perplexity Pro**

Used to find relevant documents and resources throughout the project, saving a significant amount of time that would otherwise go into manual searching. Also helpful in explaining some of the more critical and complicated concepts encountered along the way.

> **Note:** Claude Code requires an Anthropic subscription. A solid free alternative is **[Open Code](https://open-vsx.org/extension/opencode-ai/opencode)** (VS Code extension), which gives access to several free-tier models including Nvidia's **Nemotron** — capable models that work well for most coding tasks without any cost.

---

## Limitations, Reflections & Future Work

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

### AI as a Tool, Not a Crutch

AI tooling — Claude Code for development and code review, Perplexity Pro for research — works best as an aid, checker, and optimizer, not as a replacement for engineering judgement. Design decisions, architectural trade-offs, and key implementation choices still need to be owned by the developer. It helps you move faster, catch mistakes earlier, and surface better options — but every suggestion should be evaluated, and not everything it proposes will fit your actual requirements. That balance is what makes the tooling genuinely useful.
