# AI Research & Learning Companion — Project Flow

---

## What the Project Does

A personal AI tutor that:
- Accepts any file (PDF, URL, text) and makes it searchable
- Explains topics with RAG-backed answers and source citations
- Quizzes the user and tracks their score/progress
- Researches live web content and topics across uploaded documents on demand
- Remembers the user across sessions (name, progress, weak areas, style)

---

## Full Architecture

```
User message
      │
      ▼
┌─────────────────────────────────────────────────────┐
│  SUPERVISOR  (structured output → RouteDecision)    │
│  Reads: InMemoryStore user profile                  │
│  Decides: intent + topic                            │
└──────┬──────────┬──────────┬──────────┬─────────────┘
       │          │          │          │
       ▼          ▼          ▼          ▼
  [EXPLAINER]  [QUIZZER]  [RESEARCHER]  END
       │          │          │
   ChromaDB    Structured   Tavily
   RAG +        QuizSet     Search + ChromaDB + doc search +
   citations     output     summarise                      
       │          │          │
       └──────────┴──────────┘
                  │
         back to SUPERVISOR
                  │
           ┌──────┴──────┐
           │   MEMORY    │
           │ SqliteSaver │  ← session history
           │ InMemStore  │  ← user profile/progress
           │ trim_msgs   │  ← flat token cost
           └─────────────┘
```

---

## Build Strategy

Build working system first, add features on top. Never broken at any step.

```
❌ Horizontal (nothing works until everything is done)
✅ Vertical  (working demo at every step)
```

```
Phase 1 — MVP (Core working system)
  Step 1: structure + state + config
  Step 2: supervisor + explainer + ChromaDB RAG
  Step 3: memory (SqliteSaver + InMemoryStore + trim)
  → FIRST WORKING DEMO: upload PDF, ask questions, bot remembers you

Phase 2 — Add agents
  Step 4: quiz agent (structured output + score tracking)
  Step 5: planner agent (roadmap generation)
  Step 6: research agent (Wikipedia + ChromaDB doc search)
  → SECOND DEMO: full multi-agent learning companion

Phase 3 — Advanced RAG
  Step 7: Neo4j knowledge graph + entity extraction
  Step 8: graph RAG (ChromaDB + Neo4j combined)
  Step 9: knowledge agent (Neo4j queries)

Phase 4 — UI
  Step 10: Streamlit UI (file upload + chat + progress panel)
  → WORKING APP: full terminal + UI, all agents live

Phase 5 — Neo4j Knowledge Graph
  Step 11: Neo4j setup + entity extraction from docs
  Step 12: graph RAG (ChromaDB vector + Neo4j graph combined)
  Step 13: knowledge agent (Cypher queries)
  → ADVANCED DEMO: concept relationships + graph-backed answers

v2 (post-ship)
  - RAGAS evaluation (faithfulness, correctness, recall@k)
  - Docker + docker-compose (app + chromadb + neo4j)
```

Each step is runnable and testable before moving to the next.

---

## Folder Structure

```
project/
  main.py               ← entry point, while-True loop (terminal)
  config.py             ← all paths, model names, constants + SSL fix
  state.py              ← shared LearningState TypedDict
  graph.py              ← assemble the full LangGraph
  requirements.txt      ← all deps
  .env                  ← secrets (never commit)

  agents/
    supervisor.py       ← RouteDecision structured output
    explainer.py        ← RAG tutor with citations
    quizzer.py          ← QuizSet structured output + evaluator
    researcher.py       ← doc search + summariser
    planner.py          ← roadmap generator (Phase 2)
    knowledge.py        ← Neo4j graph queries (Phase 5)

  memory/
    checkpointer.py     ← SqliteSaver setup
    store.py            ← InMemoryStore setup + helpers

  vectorstore/
    ingest.py           ← load → chunk → embed → store
    retriever.py        ← load existing DB + as_retriever()

  graph_db/             ← Phase 5
    neo4j_client.py     ← Neo4j connection + Cypher queries
    entity_extractor.py ← LLM extracts entities + relationships
    graph_rag.py        ← combine ChromaDB + Neo4j

  tools/
    search.py           ← WikipediaQueryRun
    calculator.py       ← numexpr calculator tool
    code_runner.py      ← Python REPL tool (safe execution)

  evaluation/           ← v2
    ragas_eval.py       ← faithfulness, correctness, recall@k

  app.py                ← Streamlit UI (Phase 4)

  data/                 ← user uploaded files go here
  notes/                ← auto-saved explanations per topic
```

---

## Shared State

```python
class LearningState(TypedDict):
    messages:    Annotated[list[BaseMessage], add_messages]
    intent:      str        # "explain" / "quiz" / "research" / "plan" / "graph" / "finish"
    topic:       str        # what topic is being discussed
    user_id:     str        # for InMemoryStore lookups
    context:     str        # RAG retrieved chunks (for explainer)
    quiz_result: dict       # score, correct/wrong answers (for quizzer)
    graph_data:  dict       # Neo4j query results (Phase 5)
    step_count:  int        # loop guard (max iterations)
```

---

## Agents — What Each Does

### Supervisor
- Uses `llm.with_structured_output(RouteDecision)` — typed, validated routing
- Reads user profile from InMemoryStore (adjusts depth based on level)
- Returns `RouteDecision(intent, topic, reason)` — never garbage text
- Detects when task is done → FINISH

### Explainer Agent (Phase 1)
- Loads relevant chunks from ChromaDB using MMR retriever (diverse results)
- Builds personalised system prompt from InMemoryStore profile
- Returns answer WITH source citations: "From page 3 of k8s.pdf"
- Saves topic to user's progress in InMemoryStore

### Quiz Agent (Phase 2)
- Uses `llm.with_structured_output(QuizSet)` → typed MCQ questions
- Evaluates user answers with a separate LLM call
- Saves score to InMemoryStore progress
- Identifies weak areas → flags for future sessions

### Research Agent (Phase 2)
- Uses `WikipediaQueryRun` for free live web search (no API key needed)
- Also searches uploaded documents in ChromaDB for context
- Summarises findings into structured notes
- Saves summary back to ChromaDB for future retrieval

### Planner Agent (Phase 2)
- Reads user progress from InMemoryStore (completed topics, weak areas)
- Generates personalised step-by-step learning roadmap
- Example: "Create a 30-day LangGraph plan"

### Knowledge Agent (Phase 5)
- Queries Neo4j knowledge graph using Cypher
- Finds concept relationships: "What is related to Transformers?"
- Returns graph-backed answers, not just text similarity

---

## Knowledge Graph — Phase 5

### Neo4j
```
Extract from documents:
  Transformer → uses → Attention
  Attention   → part_of → LLM
  LLM         → used_in → RAG

Store in Neo4j. Query with Cypher.
```

### Graph RAG — Combine Both
```
User: "What concepts are related to Transformers?"
  → ChromaDB: semantically similar chunks (vector search)
  → Neo4j: concepts connected by edges (graph traversal)
  → LLM: combine both sources → richer, more accurate answer
```

### Knowledge Graph (Neo4j)
```
Extract from documents:
  Transformer → uses → Attention
  Attention   → part_of → LLM
  LLM         → used_in → RAG

Store in Neo4j. Query with Cypher.
```

### Graph RAG — Combine Both
```
User: "What concepts are related to Transformers?"
  → ChromaDB: semantically similar chunks (vector search)
  → Neo4j: concepts connected by edges (graph traversal)
  → LLM: combine both sources → richer, more accurate answer
```

---

## Streamlit UI — Phase 4

```
┌─────────────────────────────────────────────┐
│  AI Research & Learning Companion           │
│                                             │
│  [Upload PDF] [Upload URL] [Upload Notes]   │
│                                             │
│  ┌─────────────────────────────────────┐   │
│  │ 🤖 Hello! You've completed          │   │
│  │    RAG and LangGraph. Today's       │   │
│  │    suggestion: Knowledge Graphs     │   │
│  └─────────────────────────────────────┘   │
│                                             │
│  [Type your question...]         [Send]     │
│                                             │
│  Progress: RAG ✅  LangGraph ✅  Neo4j ⬜  │
└─────────────────────────────────────────────┘
```

Built after all agents work from terminal.
Adding UI to working agents takes ~1 hour.
Adding UI before agents work = debugging through a UI = painful.

---

## Memory — Three Layers

```
Layer 1: SqliteSaver
  → saves full conversation state per session (thread_id)
  → survives process restarts
  → used with: graph.compile(checkpointer=db)

Layer 2: InMemoryStore  (namespaces)
  ("profiles", user_id) → name, level, learning_style, session_count
  ("progress", user_id) → completed topics, quiz scores, weak areas
  ("notes",    user_id) → auto-saved explanation notes per topic

Layer 3: trim_messages
  → keeps last N messages in LLM context
  → combined with store: flat token cost forever
```

---

## Structured Output — Why It Matters

### Routing (no more free-text bugs)
```python
class RouteDecision(BaseModel):
    intent:  Literal["explain", "quiz", "research", "plan", "graph", "finish"]
    topic:   str    # e.g. "Docker containers"
    reason:  str    # why this intent was chosen

supervisor_llm = llm.with_structured_output(RouteDecision)
decision = supervisor_llm.invoke(messages)
# decision.intent is ALWAYS one of the valid values — never garbage
```

### Quiz questions (typed schema)
```python
class QuizQuestion(BaseModel):
    question:    str
    options:     list[str]   # ["A) ...", "B) ...", "C) ...", "D) ..."]
    answer:      str         # "B"
    explanation: str         # why B is correct

class QuizSet(BaseModel):
    topic:     str
    questions: list[QuizQuestion]

quiz_llm = llm.with_structured_output(QuizSet)
quiz = quiz_llm.invoke("Generate 3 MCQs on Docker containers")
# quiz.questions[0].question → clean, typed, validated — never malformed JSON
```

---

## What Makes This "Production-Grade"

| Basic (what we practised) | Project (what we'll build) |
|---|---|
| Free-text routing → breaks | `with_structured_output` → typed, validated |
| Single similarity search | MMR retriever → diverse, non-redundant results |
| `MemorySaver` | `SqliteSaver` + `InMemoryStore` |
| Hardcoded metadata | Dynamic LLM extraction |
| No citations | Source + page number in every answer |
| No progress tracking | Quiz scores + weak topics saved per user |
| One agent does everything | 5 specialised agents + typed supervisor |

---

## SSL Fix — Required on Every File

On the NetApp corporate network, add this to the top of every Python file:

```python
import os
pem = "/opt/homebrew/etc/openssl@3/certs/../cert.pem"
os.environ["REQUESTS_CA_BUNDLE"] = pem
os.environ["SSL_CERT_FILE"]      = pem
os.environ["CURL_CA_BUNDLE"]     = pem
```

Put it in `config.py` once — import config at the top of every other file → fix applied everywhere.

---

## LLM Config

Pick one provider and put it in `config.py`. Do not mix.

```python
# Option A: OpenAI via NetApp proxy (current setup)
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    model="gpt-4.1",
    base_url="https://llm-proxy-api.ai.eng.netapp.com",
    api_key=os.getenv("OPENAI_API_KEY"),
    model_kwargs={"user": "ak16683"}
)

# Option B: Groq (free, fast, no proxy dependency)
from langchain_groq import ChatGroq
llm = ChatGroq(model="llama-3.1-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
```

---

## .env Template

```bash
# LLM — pick one
OPENAI_API_KEY=your_key_here
# GROQ_API_KEY=your_key_here

# Phase 5 — Neo4j (install locally)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```

---

## requirements.txt

```
# Core — already installed ✅
langchain
langchain-core
langchain-community
langchain-openai
langchain-chroma
langchain-huggingface
langchain-text-splitters
langgraph
langgraph-checkpoint
langgraph-checkpoint-sqlite
langgraph-prebuilt
chromadb
sentence-transformers
torch
tiktoken
huggingface_hub
openai
pypdf
python-dotenv
pydantic
pydantic-settings
SQLAlchemy
streamlit
requests

# Phase 2 — install when reached ❌
wikipedia

# Phase 5 — install when reached ❌
neo4j

# v2 — install when reached ❌
ragas
numexpr
```

```bash
# Phase 2
.venv/bin/pip install wikipedia

# Phase 5
.venv/bin/pip install neo4j

# v2
.venv/bin/pip install ragas numexpr
```

---

## Step 1 Next Action

Create:
- `project/config.py` — all paths, model names, constants + SSL fix
- `project/state.py` — `LearningState` TypedDict
- `project/agents/supervisor.py` — `RouteDecision` + supervisor node
