"""Context Relevance prompts - Convert NVIDIA dual-judge templates to function format."""

import json


def context_relevance_judge1_prompt(query: str, context: str) -> str:
    """
    First judge template for context relevance evaluation.

    Args:
        query: The user's question
        context: The retrieved context to evaluate

    Returns:
        Prompt string for rating (0, 1, or 2)
    """
    safe_query = json.dumps(query)
    safe_context = json.dumps(context)

    return f"""### Instructions

You are a world class expert designed to evaluate the relevance score of a Context in order to answer the Question.
Your task is to determine if the Context contains proper information to answer the Question.
Do not rely on your previous knowledge about the Question.
Use only what is written in the Context and in the Question.
Follow the instructions below:
0. If the context does not contains any relevant information to answer the question, say 0.
1. If the context partially contains relevant information to answer the question, say 1.
2. If the context contains any relevant information to answer the question, say 2.
3. Do not penalize citation formatting differences (filename/page notation) when the context content is relevant.
4. If the question is underspecified and the context has enough information to justify a clarification request, do not force rating 0.
   In such cases, use 1 when context is partially relevant to the likely intent.
5. If context clearly lacks requested facts, this is a retrieval/coverage issue; score by actual relevance only, not by assumed external knowledge.
You must provide the relevance score of 0, 1, or 2, nothing else.
Do not explain.
Return your response as JSON in this format: {{"rating": X}} where X is 0, 1, or 2.

### Question: {safe_query}

### Context: {safe_context}

Do not try to explain.
Analyzing Context and Question, the Relevance score is """


def context_relevance_judge2_prompt(query: str, context: str) -> str:
    """
    Second judge template for context relevance evaluation.

    Args:
        query: The user's question
        context: The retrieved context to evaluate

    Returns:
        Prompt string for rating (0, 1, or 2)
    """
    safe_query = json.dumps(query)
    safe_context = json.dumps(context)

    return f"""As a specially designed expert to assess the relevance score of a given Context in relation to a Question, my task is to determine the extent to which the Context provides information necessary to answer the Question. I will rely solely on the information provided in the Context and Question, and not on any prior knowledge.

Here are the instructions I will follow:
* If the Context does not contain any relevant information to answer the Question, I will respond with a relevance score of 0.
* If the Context partially contains relevant information to answer the Question, I will respond with a relevance score of 1.
* If the Context contains any relevant information to answer the Question, I will respond with a relevance score of 2.
* Citation format mismatch alone (such as filename/page style) does not reduce relevance.
* For abrupt or underspecified questions, if Context can support a clarification-oriented response, use 1 rather than 0.
* If requested facts are missing in Context, treat this as retrieval/coverage limitation and score strictly by available relevance.
Return your response as JSON in this format: {{"rating": X}} where X is 0, 1, or 2.

### Question: {safe_query}

### Context: {safe_context}

Do not try to explain.
Based on the provided Question and Context, the Relevance score is  ["""
