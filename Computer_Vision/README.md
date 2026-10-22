# Tennis Action Classification

A PyTorch image-classification experiment using a Vision Transformer implemented in the notebook. The four-class task classifies tennis actions; it does not predict pose keypoints, despite the upstream dataset's title.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mteuslms/AI-ML-projects/blob/main/Computer_Vision/Tennis_Player_Action_Project.ipynb)

## Implementation and recorded result

The model uses 16×16 patches on 224×224 images, a learnable classification token and positional embeddings, 12 encoder blocks, 12 attention heads, and embedding size 768. Training uses AdamW, a cosine scheduler, gradient clipping and best-validation checkpoint saving over 20 epochs.

The saved run reaches **63.0% best validation accuracy** on an **80/20 random image split**. Training and validation use separate transform pipelines; only training applies random flips and rotations. The notebook includes image previews and loss/accuracy curves.

This is one validation run, not an independent test result. No pretrained-model comparison is implemented in the current notebook.

## Run

1. Open the notebook in a GPU Colab runtime.
2. Obtain the [Tennis Player Actions Dataset](https://www.kaggle.com/datasets/orvile/tennis-player-actions-dataset). The notebook uses Colab Secrets `KaggleAPIKey` and `KaggleUsername` to download it; enable access for both. Alternatively, download/unzip it manually and skip the credential/download cell.
3. Confirm `data_dir` points to `Tennis Player Actions Dataset for Human Pose Estimation/images` and inspect the class folders.
4. Run the data, architecture and training cells in order. Save `best_model_vit.pth` separately before disconnecting Colab; it is not included in the repository.

Dependency versions and the GPU model are not recorded in this notebook. A new split and model initialization are not seeded, so rerun accuracy is expected to vary. The large model/batch size may require adjusting for available GPU memory.

## Next evaluation improvements

- Fix and persist a seeded, stratified split; check duplicate or related player/sequence images across partitions.
- Compare a simple baseline and a pretrained model on the same split.
- Report class counts, confusion matrix, macro-F1 and errors on an independent test set.
- Add checkpoint reload and single-image inference examples with recorded dependencies.

Use the current result to describe the implemented experiment, not as proof of performance on unseen players or video sequences. Follow the upstream dataset's licensing and attribution requirements.
