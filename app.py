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

for root, dirs, files in os.walk("docs"):
    for filename in files:
        if filename.endswith(".txt"):
            file_path = os.path.join(root, filename)
            loader = TextLoader(file_path, encoding="utf-8")
            loaded_docs = loader.load()
            docs.extend(loaded_docs)
            print(f"Loaded: {file_path}")

print(f"Total documents loaded: {len(docs)}")


# -----------------------------
# 2. Split documents into chunks
# -----------------------------
print("\nSplitting documents into chunks...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=150
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
    search_type="mmr",
    search_kwargs={
        "k": 10,
        "fetch_k": 30
    }
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
    intent = "document_question"

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

    retrieved_docs = []

    print("\n--- Retrieved Documents Preview ---")

    for index, doc in enumerate(results, start=1):
        source = doc.metadata.get("source", "Unknown source")
        content = doc.page_content

        print(f"\nResult {index}")
        print(f"Source: {source}")
        print(content[:500])

        retrieved_docs.append(f"Source: {source}\nContent: {content}")

    print(f"\n[Retrieval Agent] Retrieved {len(retrieved_docs)} chunks.")

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

If the user asks about MR FOG products, product categories, flavors, devices, specifications, wholesale, distributor information, contact information, warranty, returns, or FAQ, answer using the retrieved MR FOG website context.

If the retrieved context contains related MR FOG information, summarize it clearly.
Only say "I do not have enough information in the company documents" when the retrieved context has no relevant information at all.

At the end of your answer, mention the source document name from the context.

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
    return "retrieve"

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

agent_app = workflow.compile()


# -----------------------------
# 12. Function for UI or CLI
# -----------------------------
def ask_question(user_question: str):
    result = agent_app.invoke({
        "question": user_question,
        "intent": "",
        "documents": [],
        "answer": "",
        "grounded": False
    })

    return result


# -----------------------------
# 13. Run Chatbot in Terminal
# -----------------------------
if __name__ == "__main__":
    while True:
        user_question = input("\nUser Question: ")

        if user_question.lower() in ["exit", "quit"]:
            print("Goodbye.")
            break

        result = ask_question(user_question)

        print("\nFinal Answer:")
        print(result["answer"])

        print("\nGrounded:")
        print(result["grounded"])