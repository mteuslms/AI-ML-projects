# Recorded aggregate results

`recorded_summary.csv` transcribes the section 10 output saved in `Socratic_Tutor.ipynb` on the main-branch snapshot `7f269a7b3787e89647ba29a3c541476e1ec66aba`. It does not represent another training or evaluation run.

Each condition has 35 scored responses and no missing judge calls. Strict successes are responses scoring 2. Normalized score divides rubric points by twice the number of scored responses. Percentages are stored without presentation rounding.

The source notebook reports 468 unique preferences (421 training / 47 validation), seed 42, one DPO epoch, rank-16 LoRA, Llama 3 8B in 4-bit, and GPT-4o judging. No per-prompt replies/scores, original JSONL dataset, training manifest or adapter files are available in this repository snapshot. Do not infer category scores, statistical significance, hardware, or judge correctness from this aggregate file.

To make a future run auditable, publish permissible `eval_results.json`, selected response comparisons, and a runtime/training manifest in a separately named results folder. Inspect them for sensitive content first. Keep credentials and model binaries out of Git. Do not replace this historical summary with metrics from a different run without changing its provenance.
