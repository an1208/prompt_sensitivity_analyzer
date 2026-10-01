# Prompt Sensitivity Report

## Configuration

- **backend**: ollama:llama3.2:3b
- **temperature**: 0.7
- **n_samples**: 3
- **seed**: 0
- **max_new_tokens**: 64
- **prompt_file**: C:\Users\anush\Downloads\prompt-sensitivity-analyser\prompt-sensitivity-analyser_project\data\prompt_sets.json
- **groups**: 8
- **started**: 2026-09-30T14:42:45
- **embedder**: sentence-transformers:sentence-transformers/all-MiniLM-L6-v2

## Overall

| Metric | Value |
|---|---|
| Groups analysed | 8 |
| Mean sensitivity (1 - cosine) | 0.125  (95% bootstrap CI 0.082 - 0.176) |
| Median / max sensitivity | 0.118 / 0.257 |
| Mean cross-paraphrase output similarity | 0.875 |
| Mean within-paraphrase similarity (sampling noise) | 0.924 |
| Mean excess sensitivity (within - between) | 0.049 |
| Mean lexical (Jaccard) similarity | 0.514 |
| Mean exact-match rate across paraphrases | 0.022 |
| Prompt-sim vs output-sim Spearman (pooled) | 0.339 |
| Mean reference-similarity spread | 0.088 |

## Per-group results (most sensitive first)

| Group | Category | Sensitivity | Min pair sim | Within sim | Excess | Lexical sim | Exact match | Fragile variant |
|---|---|---|---|---|---|---|---|---|
| rag_leave | rag_style | 0.257 | 0.593 | 0.845 | 0.102 | 0.417 | 0.00 | v2 |
| advice_sleep | open_ended | 0.196 | 0.671 | 0.899 | 0.095 | 0.367 | 0.00 | v3 |
| fact_photosynthesis | factual | 0.151 | 0.701 | 0.950 | 0.101 | 0.426 | 0.00 | v4 |
| rag_refund | rag_style | 0.124 | 0.821 | 0.891 | 0.016 | 0.374 | 0.00 | v1 |
| reason_apples | reasoning | 0.113 | 0.858 | 0.902 | 0.014 | 0.473 | 0.00 | v4 |
| reason_train | reasoning | 0.066 | 0.885 | 0.982 | 0.047 | 0.560 | 0.00 | v3 |
| fact_capital | factual | 0.051 | 0.890 | 0.958 | 0.009 | 0.708 | 0.12 | v1 |
| summarise_bike_lanes | summarisation | 0.043 | 0.943 | 0.963 | 0.006 | 0.785 | 0.06 | v2 |

## Most fragile prompts

### rag_leave — sensitivity 0.257

| Var | Mean sim to others | Prompt | Output (sample 0) |
|---|---|---|---|
| v0 | 0.758 | Context: Employees receive 20 days of paid leave per year. Up to 5 un… | According to the question, up to 5 unused leave days can be carried f… |
| v1 | 0.753 | Context: Employees receive 20 days of paid leave per year. Up to 5 un… | The carry-forward limit for unused leave is 5 days. |
| v2 | 0.695 | Context: Employees receive 20 days of paid leave per year. Up to 5 un… | According to the context, yes, you can carry over leave that you did … |
| v3 | 0.700 | Using only the context below, answer the question. ⏎ Context: Employe… | According to the context, up to 5 unused leave days can be carried fo… |
| v4 | 0.810 | Question: How many unused leave days can be carried forward? ⏎ Contex… | According to the information provided, up to 5 unused leave days can … |

### advice_sleep — sensitivity 0.196

| Var | Mean sim to others | Prompt | Output (sample 0) |
|---|---|---|---|
| v0 | 0.807 | How can I improve my sleep quality? | Improving sleep quality can have a significant impact on both physica… |
| v1 | 0.815 | Give me tips for sleeping better. | Here are some tips for sleeping better: ⏎  ⏎ 1. **Establish a bedtime… |
| v2 | 0.834 | What should I do to get better sleep at night? | Improving sleep quality can have a significant impact on your overall… |
| v3 | 0.736 | I have trouble sleeping. Any advice? | I'm so sorry to hear that you're struggling with sleep. Here are some… |
| v4 | 0.829 | List ways to improve the quality of my sleep. | Here are some ways to improve the quality of your sleep: ⏎  ⏎ 1. **Es… |

### fact_photosynthesis — sensitivity 0.151

| Var | Mean sim to others | Prompt | Output (sample 0) |
|---|---|---|---|
| v0 | 0.891 | What is photosynthesis? | Photosynthesis is the process by which plants, algae, and some bacter… |
| v1 | 0.882 | Explain photosynthesis. | Photosynthesis is the process by which plants, algae, and some bacter… |
| v2 | 0.862 | Can you tell me what photosynthesis is? | Photosynthesis is a fascinating process that occurs in plants, algae,… |
| v3 | 0.884 | Describe the process of photosynthesis in plants. | Photosynthesis is the process by which plants, algae, and some bacter… |
| v4 | 0.725 | In simple terms, how do plants make food from sunlight? | Plants make food from sunlight through a process called photosynthesi… |

## Figures

![sensitivity_by_group.png](sensitivity_by_group.png)
![heatmaps.png](heatmaps.png)

## How to read this

* **Sensitivity** is `1 - mean cosine similarity` between output embeddings of *different* paraphrases. 0 means outputs are semantically identical regardless of wording; larger means wording changes the answer.
* **Within sim / Excess** separate wording effects from sampling randomness. Run with `--temperature 0.7 --n-samples 3` (or more) to populate them: if paraphrase outputs are no less similar than repeated samples of one prompt (excess ≈ 0), the model is robust.
* **Prompt-sim vs output-sim Spearman** > 0 means prompts that are worded more differently produce more different outputs.
* Embedding similarity measures *semantic closeness of outputs*, not correctness. Add a `reference` answer per group to also get `reference_sim_*` (does wording change how close the answer is to the gold answer?).
* Absolute values depend on the embedder; compare models/prompts using the **same** embedder and dataset.
