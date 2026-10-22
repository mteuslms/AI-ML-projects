# AI and Machine Learning Portfolio

**Mateus Silvestre** · M.S. Artificial Intelligence student, University of South Florida

Experiments in LLM preference alignment, computer vision, and neural text classification. The featured projects include implementation, saved training/evaluation outputs, and explicit limitations. Smaller architecture exercises and course scaffolds are listed separately below.

## Featured projects

| Project | My implementation | Recorded result | Explore |
| --- | --- | --- | --- |
| **Socratic Tutor** | 4-bit Llama 3 8B with rank-16 LoRA and DPO; four adapter/system-prompt conditions | Strict adherence **97.1% with DPO + prompting vs. 88.6% with prompting alone**, on 35 prompts. Both score 5.7% without prompting. | [Project details](Advanced_Reasoning/README.md) · [Executed notebook](Advanced_Reasoning/Socratic_Tutor.ipynb) |
| **Tennis Action Classification** | 12-layer, 12-head Vision Transformer in PyTorch; four classes, augmentation, AdamW and cosine scheduling | **63.0% validation accuracy** on an 80/20 random split after 20 epochs | [Project details](Computer_Vision/README.md) · [Executed notebook](Computer_Vision/Tennis_Player_Action_Project.ipynb) |

These are recorded single-run validation/evaluation results, not independently reproduced benchmarks. Tutor adherence measures a response-style policy, not student learning; the tennis model has no independent test score or pretrained comparison in the current notebook.

## Supporting NLP project

**[SMS Spam Classification](LLMs_and_Transformers/Spam_or_Ham.ipynb)** uses a learned embedding and MLP with FastAI/PyTorch. Recorded validation accuracy is **97.0%** on an 80/20 split of 5,572 SMS messages. Precision, recall and a separate test set are needed before broader performance claims. [NLP project notes](LLMs_and_Transformers/README.md)

## Learning exercises and coursework

- **[TinyGPT](LLMs_and_Transformers/PracticeTinyGPT.ipynb):** small causal single-head attention model with sinusoidal positional encodings and toy next-token generation. This is a fundamentals exercise; padding handling remains a limitation.
- **[Transformer Build](LLMs_and_Transformers/Transformer_Build.ipynb):** minimal single-head attention/residual-block demonstration on four toy examples, without positional encoding or a causal mask.
- **[CamVid segmentation coursework](coursework/README.md):** U-Net/ResNet-34 course exercise. The current Dice output and local image paths need correction; this is not a validated segmentation benchmark.

## Run and inspect

The executed notebooks can be viewed on GitHub without a runtime. Use the Colab links in each project README to run code; choose a GPU runtime for the larger models. Saved outputs document prior runs and do not guarantee an identical rerun.

- Tutor training requires your own preference JSONL or explicitly enabled paid generation. API judging is optional, paid and disabled by default. The full original dataset and adapter weights are not included.
- Tennis data is downloaded separately from Kaggle; follow the notebook's secret setup or supply the dataset manually.
- SMS data is loaded from the URL in the notebook. Other than the Tutor's pinned core packages, these notebooks still need recorded dependency versions for reproducible environments.
- Saving a notebook to GitHub does not save its runtime model files. Keep model weights, credentials and private data outside Git.

## Offline checks

From the repository root, using Python's standard library:

```bash
python -m unittest discover -s Advanced_Reasoning/tests -v
```

The checks cover Tutor Python syntax, absence of stored error outputs, public configuration defaults, data and score validation, metric denominators, and aggregate result consistency. They do not run GPU training or external APIs.

## Tools used

Python, PyTorch, Transformers, PEFT, TRL, bitsandbytes, FastAI, Pandas, Matplotlib, Git, and Google Colab. The Tutor currently uses Transformers/PEFT QLoRA; the `unsloth/` model identifier names the upstream quantized weights, not an Unsloth training dependency.

Upstream models and datasets retain their own licenses. Course material is identified as coursework; these notebooks should not be interpreted as wholly original implementations of the accompanying course scaffolds.
