"""

    psa run --backend callable --model examples/my_rag.py:answer --prompts data/prompt_sets.json

Replace the body of `answer` with a call into your pipeline (retrieve -> generate) and return the final
answer text. The analyser then reports how much your *whole pipeline's* answers change when only the
wording of the question changes - retrieval variance included.
"""


def answer(prompt: str) -> str:
    # from my_rag_project.pipeline import ask   # <- your code here
    # return ask(prompt)
    return "This is a placeholder answer for: " + prompt.splitlines()[-1][:60]
