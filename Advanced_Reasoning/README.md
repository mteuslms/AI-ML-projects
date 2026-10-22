# Socratic Tutor

A QLoRA + Direct Preference Optimization experiment that trains Llama 3 to offer specific guiding questions instead of finished answers. It compares the same quantized base model with its adapter enabled/disabled, each with and without a Socratic system prompt.

## Run

Open `Socratic_Tutor.ipynb` in a fresh Python 3.11 CUDA GPU runtime. Run the installation cell and restart the runtime, then run the remaining cells in order after configuring the notebook. The default `MODE = "inspect"` skips training and model loading. Dependencies use a fixed Transformers/TRL/PEFT API baseline; the notebook records resolved training package versions. The complete dependency combination still needs GPU validation.

- **Train:** set `MODE = "train"` and supply `socratic_data.jsonl` in the runtime working directory. Each line must contain nonempty `prompt`, `chosen`, and `rejected` strings. At least 10 distinct prompts are required; roughly 500 examples matches the original experiment target. Review preference correctness manually.
- **Generate data:** optionally set `GENERATE_DATA = True` and supply `OPENAI_API_KEY` through an environment variable or Colab Secrets. This makes paid API requests and targets 500 examples. Existing data is never overwritten; partial files require manual inspection before retrying.
- **Load:** set `MODE = "load"` and point `ADAPTER_DIR` at an existing PEFT adapter, including its tokenizer files. Training data is unnecessary in this mode.
- **Evaluate:** train/load modes generate 140 responses across 35 retained prompts and four conditions. Set `RUN_API_JUDGE = True` only when ready for paid API judging. Generation and judging are separate; judge-only reruns can use inspect mode with saved responses. Missing scores are retried when running the judge cell again in the same session. Across restarts, load `eval_results.json` into `results` before that cell to preserve scored records.

The dataset and trained weights are **not included**. The original uploaded notebook referenced a private Drive dataset; it cannot be recovered from notebook code. Mount Drive separately and change paths if you want Colab persistence. Keep model files outside Git. An 8B QLoRA workload may exceed T4 memory; reduce sequence lengths (keeping room for completions), or use a smaller compatible instruct model through `BASE_MODEL`. No hardware-specific memory or speed guarantee is made.

## Outputs

`socratic_tutor_run/` contains the adapter, tokenizer, split prompts, dataset SHA-256, training configuration and package versions, training history, generated responses, judge results, evaluation metadata, overall/category summaries, and an adherence plot when scores exist. These files contain generated experiment data and are ignored by Git. Save them separately before ending an ephemeral runtime.

Strict Socratic Adherence Rate counts only score-2 responses. The normalized rubric score gives half credit for score 1. Both exclude failed judge calls and report coverage; the notebook also reports matched-prompt differences to reduce unequal-coverage confusion.

## Validation

From the repository root:

```bash
python -m unittest discover -s Advanced_Reasoning/tests -v
```

These offline checks cover notebook Python syntax, clean outputs, preference validation/deduplication, judge-score validation, missing-score denominators, partial-credit metrics, and the 35-prompt inventory. GPU training, model downloads, API judging, and learning outcomes are not verified by these checks.

## Research limits

This is a small exploratory response-style experiment, not evidence of improved student learning. Repeated concepts, underspecified evaluation questions, synthetic preferences, one seed, and a single automated judge limit conclusions. Review exact and semantic train/evaluation overlap; load mode cannot check overlap without the original training data. Use blinded human review and repeated runs before making stronger claims. The same generator/judge family may introduce bias. Ordinary translation or factual assistance can fail this project's strict tutoring rubric while remaining useful assistance.

The standard Transformers/PEFT stack replaces the original Unsloth patch and unsafe `--no-deps` installation sequence. Conversational DPO data delegates special-token formatting to TRL. The supported model output is a PEFT adapter; GGUF conversion is outside the default workflow. Use of the base weights remains subject to their upstream model license.

API reference: [TRL 0.24 DPOTrainer](https://github.com/huggingface/trl/blob/v0.24.0/trl/trainer/dpo_trainer.py).
