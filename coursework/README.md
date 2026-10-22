# Computer Vision Coursework

[CamVid segmentation notebook](imageSemanticSegmentation_revised_2_14_2026.ipynb) · [Open in Colab](https://colab.research.google.com/github/mteuslms/AI-ML-projects/blob/main/coursework/imageSemanticSegmentation_revised_2_14_2026.ipynb)

A course-scaffold exercise using FastAI U-Net with a pretrained ResNet-34 backbone on CamVid Tiny. The task prompts and cells marked as provided code are retained as course material; the notebook is listed separately from the featured portfolio experiments.

## Current limitations

- The saved `Dice()` values are around 15-17 and are not valid on the usual 0-1 Dice scale. Verify a multiclass Dice/mIoU metric and label handling before reporting performance.
- The later frozen fit continues from the earlier fine-tuned model; it is not an independent baseline comparison.
- A local image path differs from the downloaded file. Standard PIL images use `size`, not `shape`.
- The final custom-image task is not completed in the saved notebook.

This notebook needs those corrections and a clean rerun before being used as a validated segmentation example. Its current score should not be cited as model performance. Retain course and upstream dataset attribution when revising it.
