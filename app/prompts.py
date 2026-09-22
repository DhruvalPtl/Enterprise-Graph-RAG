"""
Prompt Templates and System Instructions for Grounded Enterprise RAG (Step 6).

Defines strict guidelines for hallucination-resistant, citation-grounded generation:
1. Answers must be derived exclusively from the provided retrieved context.
2. If context is insufficient, model must explicitly state lack of information.
3. Model references evidence using stable [SOURCE X] tags.
4. Model never fabricates metadata, documents, or citations.
"""

NO_EVIDENCE_MESSAGE = "The available documents do not provide sufficient information to answer this question."

SYSTEM_INSTRUCTION = """You are an expert enterprise knowledge assistant.

Your primary directive is to provide accurate, grounded answers strictly based on the provided retrieved context passages.

STRICT INSTRUCTIONS:
1. Answer using ONLY the facts explicitly stated in the RETRIEVED CONTEXT below.
2. Do NOT invent, assume, extrapolate, or bring in outside knowledge that is not directly supported by the context.
3. If the retrieved context does not contain enough information to answer the question, state:
   "The available documents do not provide sufficient information to answer this question."
   Do NOT guess or attempt to partially make up an answer.
4. Distinguish uncertainty clearly rather than guessing.
5. Refer to specific evidence using the exact source identifiers provided in the context, such as [SOURCE 1] or [SOURCE 2].
6. Do NOT fabricate document names, page numbers, section names, or citation numbers.
7. Answer the user's question directly, clearly, and concisely in professional markdown format.
"""


def build_rag_prompt(query: str, context_text: str) -> str:
    """
    Constructs the complete user prompt, cleanly separating retrieved context
    and the user question.

    Args:
        query: User question or search query.
        context_text: Structured context text formatted by ContextBuilder.

    Returns:
        Formatted prompt string.
    """
    clean_query = query.strip()
    clean_context = context_text.strip() if context_text else "No relevant context found."

    return f"""RETRIEVED CONTEXT:
================================================================================
{clean_context}
================================================================================

USER QUESTION:
{clean_query}

ANSWER (Grounded strictly in the retrieved context above, with [SOURCE X] citations):"""
