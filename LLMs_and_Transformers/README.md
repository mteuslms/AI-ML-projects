# NLP Models and Transformer Fundamentals

This folder combines a small applied text classifier with architecture learning exercises. The exercises demonstrate tensor operations and training mechanics; they are not implementations of production-scale GPT or Llama.

## SMS spam classification

[Notebook](Spam_or_Ham.ipynb) · [Open in Colab](https://colab.research.google.com/github/mteuslms/AI-ML-projects/blob/main/LLMs_and_Transformers/Spam_or_Ham.ipynb)

A FastAI/PyTorch classifier using learned word embeddings, pooled features and an MLP. The notebook loads 5,572 SMS messages (4,825 ham; 747 spam) from the linked dataset URL and uses an 80/20 train/validation split with split seed 17.

The recorded final validation accuracy is **97.04%** after 10 epochs; training accuracy is 99.23%. These are validation results, not an independent test score. The ten hand-authored prediction examples are demonstrations, not evidence of general reliability. Predicted ham does not establish that a message is safe.

To rerun, provide a compatible Python/PyTorch/FastAI runtime and run cells in order; the dataset downloads from the URL in the notebook. Dependency versions are not pinned. Before broader performance claims, deduplicate data before splitting, add a simple TF-IDF baseline, measure spam precision/recall/F1 and PR-AUC, and test preprocessing parity for the custom inference path. Mask padding when pooling variable-length messages.

## TinyGPT fundamentals

[Notebook](PracticeTinyGPT.ipynb) · [Open in Colab](https://colab.research.google.com/github/mteuslms/AI-ML-projects/blob/main/LLMs_and_Transformers/PracticeTinyGPT.ipynb)

A toy next-token model with single-head causal self-attention, sinusoidal positional encodings, feed-forward layers and residual connections. It learns a few short token sequences and demonstrates greedy decoding. Its residual normalization is not standard pre-norm, and it does not implement multi-head attention.

The saved output includes repeated padding tokens. Padding targets are currently included in the loss. Fix padding/EOS handling and add causality tests before expanding the experiment. Defining a CUDA device alone does not move this notebook's model/data to GPU. Run it as a small educational CPU example with PyTorch installed; it does not demonstrate a held-out language-model benchmark.

## Minimal transformer block

[Notebook](Transformer_Build.ipynb) · [Open in Colab](https://colab.research.google.com/github/mteuslms/AI-ML-projects/blob/main/LLMs_and_Transformers/Transformer_Build.ipynb)

A small embedding, single-head attention, MLP and pre-norm residual-block demonstration trained on four toy examples. It has no positional encoding or causal mask. The printed next-token example illustrates fitting toy data; it is not evidence of general language understanding. A PyTorch CPU environment is sufficient for its scale.

Neither architecture exercise demonstrates a pretrained Llama-logit comparison or a general multi-head GPT implementation. Preserve attribution for any course or tutorial material when extending these notebooks.
