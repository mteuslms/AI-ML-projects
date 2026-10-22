# Socratic Tutor

A QLoRA and Direct Preference Optimization experiment that encourages Llama 3 to offer guiding questions instead of final answers. It compares one quantized base model with its LoRA adapter enabled/disabled, with and without a Socratic system prompt.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mteuslms/AI-ML-projects/blob/main/Advanced_Reasoning/Socratic_Tutor.ipynb)

## Recorded results

The executed notebook contains one completed run: **468 unique preference examples**, split into **421 training / 47 validation** examples; rank-16 LoRA, one DPO epoch, learning rate `5e-5`, beta `0.1`, and seed 42. The synthetic data generator targets 500 examples; that target is not the number used after loading/deduplication in this run.

Evaluation uses 35 hand-authored prompts across emotional, authority, context and stealth categories. GPT-4o scored all 140 responses across the four conditions.

| Condition | Strict successes | Strict adherence | Normalized rubric score | Judge coverage |
| --- | --- | --- | --- | --- |
| Base + system prompt | 31/35 | 88.6% | 90.0% | 100% |
| DPO + system prompt | 34/35 | 97.1% | 97.1% | 100% |
| Base without system prompt | 2/35 | 5.7% | 7.1% | 100% |
| DPO without system prompt | 2/35 | 5.7% | 10.0% | 100% |

**Interpretation:** DPO plus prompting improved strict adherence by 8.6 percentage points, or three prompts, over prompting alone in this run. Without a system prompt, strict adherence did not improve. These results do not establish prompt-independent alignment, statistical significance, or improved student learning.

[Aggregate results CSV](results/recorded_summary.csv) · [Result provenance](results/README.md) · [Executed notebook](Socratic_Tutor.ipynb)

The CSV transcribes the saved notebook output. Per-response judge records, the original preference dataset, a hardware/runtime manifest, and model weights are not committed. A reviewer can inspect the training code and aggregate output but cannot audit every judgement or exactly reproduce this historical run from the repository alone.

## Run a new experiment in Colab

1. Open the notebook and select a CUDA GPU runtime. The saved run reported Python 3.13 in its package paths; record your own Python/GPU/package versions. Other environments are not guaranteed compatible.
2. Run the installation cell, restart the runtime, and continue with configuration. The public defaults are `MODE = "inspect"`, `GENERATE_DATA = False`, and `RUN_API_JUDGE = False`. Historical outputs remain visible even though defaults have been reset.
3. Upload `socratic_data.jsonl` to the runtime working directory and set `MODE = "train"`. Each JSONL row must contain nonempty `prompt`, `chosen`, and `rejected` strings. At least 10 unique prompts are required; manually review preference quality. Alternatively, set `MODE = "load"` and point `ADAPTER_DIR` at an existing adapter with its tokenizer files.
4. With API judging disabled, run configuration through **section 8** to train/load and generate evaluation responses. Training does not require an OpenAI key when data is supplied.
5. To judge, add `OPENAI_API_KEY` to Colab Secrets and enable Notebook access. Run `RUN_API_JUDGE = True` in a **new cell**, then run **sections 9 and 10 only**. This makes paid requests; do not rerun training or response generation. The legacy secret name `OpenAI_API_Key` is also accepted.
6. Inspect coverage, response quality and charts. Persist or download `socratic_tutor_run/` before the runtime disconnects. Saving the notebook to GitHub does not save the adapter or separate JSON files.

Optional synthetic generation uses `GENERATE_DATA = True` and paid API calls. It refuses to overwrite an existing dataset and keeps partial output separately on failure. The original dataset cannot be reconstructed from the notebook alone; regenerated data forms a different experiment.

Drive is optional: mount it separately and change `WORK_DIR`/`DATA_PATH` if you want persistent storage. Keep binary weights out of Git. An 8B QLoRA workload may exceed T4 memory; reduce sequence lengths or use a smaller compatible instruct model. No hardware-specific fit or speed guarantee is claimed.

## Resume interrupted judging

Restore the run folder, retain `MODE = "inspect"`, and run the setup/configuration/helper cells. Skip GPU loading/training. Set `RUN_API_JUDGE = True` and run section 9, which loads `eval_results.json` if available, otherwise `eval_responses.json`. Only missing scores are judged. Run section 10 for summaries. If you already have `results` in memory, that current in-memory run takes precedence.

## Outputs and metrics

`socratic_tutor_run/` contains the adapter/tokenizer, split prompts, dataset hash, training configuration and package versions, training history, generated responses, judge results, evaluation metadata, category summaries and plot. It is ignored by Git. Publish only reviewed, permissible result records in a separately named `results/` subfolder; do not publish private data or credentials.

- **Strict SAR:** score-2 responses / successfully scored responses.
- **Normalized rubric score:** total points / (2 × successfully scored responses). Score 1 receives partial credit.
- **Coverage:** successfully scored / all responses. Failed judge calls remain missing rather than counting as model failures.
- **Paired difference:** compares prompts scored in both conditions, reducing unequal-coverage confusion.

## Offline validation

```bash
python -m unittest discover -s Advanced_Reasoning/tests -v
```

Run this from the repository root. Checks accept published outputs but reject stored error outputs, verify Python syntax and opt-in defaults, and exercise data validation/deduplication, judge-score validation, missing-score denominators, partial credit, the 35-prompt inventory and summary consistency. They do not rerun training or validate judge correctness. The completed GPU/API run is evidenced by the saved notebook outputs, not by these tests.

## Research limits and implementation

The evaluation is small, contains repeated concepts and some underspecified questions, and uses one training seed and a single automated judge. Data generation and judging use GPT-4o, which can introduce correlated preferences. Review semantic train/evaluation overlap, blind-review outputs with human raters, and repeat across seeds before stronger conclusions. Strictly withholding answers can also penalize useful factual or translation assistance.

Transformers/PEFT QLoRA replaces the earlier Unsloth patches. Conversational DPO data delegates special-token formatting to TRL. Inference uses CUDA autocast to handle mixed model dtypes, including when adapters are disabled. PEFT adapter export is supported; GGUF conversion is outside this workflow. Upstream base-model licensing still applies.

API reference: [TRL 0.24 DPOTrainer](https://github.com/huggingface/trl/blob/v0.24.0/trl/trainer/dpo_trainer.py).
