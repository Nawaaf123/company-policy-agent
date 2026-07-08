import os
from typing import TypedDict, List
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma

from langgraph.graph import StateGraph, START, END


load_dotenv()


# -----------------------------
# 1. Load company documents
# -----------------------------
print("Loading company documents...")

docs = []

for filename in os.listdir("docs"):
    if filename.endswith(".txt"):
        file_path = os.path.join("docs", filename)
        loader = TextLoader(file_path, encoding="utf-8")
        loaded_docs = loader.load()
        docs.extend(loaded_docs)
        print(f"Loaded: {filename}")

print(f"Total documents loaded: {len(docs)}")


# -----------------------------
# 2. Split documents into chunks
# -----------------------------
print("\nSplitting documents into chunks...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(docs)

print(f"Total chunks created: {len(chunks)}")


# -----------------------------
# 3. Create embeddings and vector database
# -----------------------------
print("\nCreating embeddings and vector database...")

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="company_policy_docs"
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

print("Vector database ready.")


# -----------------------------
# 4. Load LLM
# -----------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

print("LLM ready.")


# -----------------------------
# 5. Define Agent State
# -----------------------------
class AgentState(TypedDict):
    question: str
    intent: str
    documents: List[str]
    answer: str
    grounded: bool


# -----------------------------
# 6. Supervisor Agent
# -----------------------------
def supervisor_agent(state: AgentState):
    question = state["question"].lower()

    policy_keywords = [
        "policy",
        "reimbursement",
        "travel",
        "remote",
        "laptop",
        "contractor",
        "employee",
        "work from home"
    ]

    if any(word in question for word in policy_keywords):
        intent = "document_question"
    else:
        intent = "general_question"

    print(f"\n[Supervisor Agent] Intent detected: {intent}")

    return {
        "intent": intent
    }


# -----------------------------
# 7. Retrieval Agent
# -----------------------------
def retrieval_agent(state: AgentState):
    print("[Retrieval Agent] Searching company documents...")

    results = retriever.invoke(state["question"])

    retrieved_docs = [doc.page_content for doc in results]

    print(f"[Retrieval Agent] Retrieved {len(retrieved_docs)} chunks.")

    return {
        "documents": retrieved_docs
    }


# -----------------------------
# 8. Answer Agent
# -----------------------------
def answer_agent(state: AgentState):
    print("[Answer Agent] Generating final answer...")

    context = "\n\n".join(state.get("documents", []))

    prompt = f"""
You are an enterprise company knowledge assistant.

Answer the user's question only using the provided company context.
Do not make up information.

If the answer is not available in the company context, say:
"I do not have enough information in the company documents."

User question:
{state["question"]}

Company context:
{context}
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }


# -----------------------------
# 9. Guardrail Agent
# -----------------------------
def guardrail_agent(state: AgentState):
    print("[Guardrail Agent] Checking answer grounding...")

    documents = state.get("documents", [])
    answer = state.get("answer", "")

    if state["intent"] == "document_question" and len(documents) == 0:
        return {
            "grounded": False,
            "answer": "I do not have enough information in the company documents."
        }

    if "I do not have enough information" in answer:
        return {
            "grounded": False
        }

    return {
        "grounded": True
    }


# -----------------------------
# 10. Routing Function
# -----------------------------
def route_after_supervisor(state: AgentState):
    if state["intent"] == "document_question":
        return "retrieve"
    else:
        return "answer"


# -----------------------------
# 11. Build LangGraph Workflow
# -----------------------------
workflow = StateGraph(AgentState)

workflow.add_node("supervisor", supervisor_agent)
workflow.add_node("retrieve", retrieval_agent)
workflow.add_node("answer", answer_agent)
workflow.add_node("guardrail", guardrail_agent)

workflow.add_edge(START, "supervisor")

workflow.add_conditional_edges(
    "supervisor",
    route_after_supervisor,
    {
        "retrieve": "retrieve",
        "answer": "answer"
    }
)

workflow.add_edge("retrieve", "answer")
workflow.add_edge("answer", "guardrail")
workflow.add_edge("guardrail", END)

app = workflow.compile()


# -----------------------------
# 12. Run Chatbot
# -----------------------------
while True:
    user_question = input("\nUser Question: ")

    if user_question.lower() in ["exit", "quit"]:
        print("Goodbye.")
        break

    result = app.invoke({
        "question": user_question,
        "intent": "",
        "documents": [],
        "answer": "",
        "grounded": False
    })

    print("\nFinal Answer:")
    print(result["answer"])

    print("\nGrounded:")
    print(result["grounded"])