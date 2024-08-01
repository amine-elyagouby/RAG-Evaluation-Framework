# RAG-Evaluation-Framework

## Evaluation Results

You can skip the experimentation for the evaluation of IR and LLM and find the results here:

[Evaluation Results](https://drive.google.com/drive/folders/1pgUPmpIdepVifLUubNuT5mInsj2hkX3v)

This folder contains:

- `LLM_scores.json`: Evaluation using GPT-4 of the responses.
- `ir_scores`: Folder containing JSON files of the IR scores calculated for the 5 retrievals at each level of N number of tokens, with N being 100, 200, and 1000.
- `IR_LLM_Evaluations_Analysis.ipynb`: Jupyter notebook with the IR and LLM evaluation results and the analysis of their relationship.




## Illustrations

### Illustration of the study conducted to evaluate Retrieval-Augmented Generation (RAG) systems

![Illustration of Study](https://github.com/user-attachments/assets/eeb31faf-cf0d-440d-b486-5d3955399632)

### Illustration of our evaluation framework

It begins by assessing the Information Retrieval (IR) system performance and directly estimates the overall RAG performance from it, eliminating the need for expensive response generation and evaluation.

![Evaluation Framework](https://github.com/user-attachments/assets/5e99bdcc-5e0b-4186-9078-bea1c1e3f2bb)

