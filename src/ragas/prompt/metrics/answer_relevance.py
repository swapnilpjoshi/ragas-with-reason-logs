"""Answer Relevance prompt for generating questions and detecting noncommittal responses."""

import json


def answer_relevancy_prompt(response: str) -> str:
    """
    Generate the prompt for answer relevance evaluation.

    Args:
        response: The response text to evaluate

    Returns:
        Formatted prompt string for the LLM
    """
    # Use json.dumps() to safely escape the response string
    safe_response = json.dumps(response)

    return f"""Generate a question for the given answer and identify if the answer is noncommittal.
Give noncommittal as 1 if the answer is noncommittal and 0 if the answer is committal.

Definition:
- Noncommittal (1): evasive, vague, generic refusal without grounding, or does not make a concrete claim.
- Committal (0): directly answers, OR gives a concrete evidence-aware refusal such as:
  "the provided documents/context do not contain that data", "no matching information found in the supplied files", with clear scope.
  Also committal when the response is a valid clarification request for an underspecified question.

Important:
- Do not mark an answer as noncommittal only because it says data is unavailable, if it clearly states this is based on provided evidence/context.
- Mark as noncommittal when the response only avoids commitment (e.g., "I don't know", "I'm not sure", "please check manual") without evidence-based scope.
- If the user question is abrupt/underspecified (e.g., only product name, "test", very short fragment), a clarification request is a valid committal response.
- If the message looks like a follow-up turn that depends on prior conversation, do not mark noncommittal only because the answer asks for missing specifics.

--------EXAMPLES-----------
Example 1
Input: {{
    "response": "Albert Einstein was born in Germany."
}}
Output: {{
    "question": "Where was Albert Einstein born?",
    "noncommittal": 0
}}

Example 2
Input: {{
    "response": "I don't know about the  groundbreaking feature of the smartphone invented in 2023 as am unaware of information beyond 2022. "
}}
Output: {{
    "question": "What was the groundbreaking feature of the smartphone invented in 2023?",
    "noncommittal": 1
}}

Example 3
Input: {{
    "response": "I checked the provided documents and no entry for RZ/A3UL was found in that dataset."
}}
Output: {{
    "question": "Was RZ/A3UL found in the provided documents?",
    "noncommittal": 0
}}

Example 4
Input: {{
    "response": "The question is underspecified. Please provide the exact device, interface, and condition so I can answer precisely."
}}
Output: {{
    "question": "What additional details are needed to answer the underspecified technical question?",
    "noncommittal": 0
}}
-----------------------------

Now perform the same with the following input
input: {{
    "response": {safe_response}
}}
Output: """
