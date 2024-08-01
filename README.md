# RAG-Evaluation-Framework

## Evaluation Results

You can skip the experimentation for the evaluation of IR and LLM and find the results here:

[Evaluation Results](https://drive.google.com/drive/folders/1pgUPmpIdepVifLUubNuT5mInsj2hkX3v)

This folder contains:

- `LLM_scores.json`: Evaluation results (scores) using GPT-4 of the responses.
- `ir_scores`: Folder containing 5 JSON files of the IR scores calculated for the 5 retrievals at each level of N number of tokens, with N in {100, 200,..., 1000}.


`IR_LLM_Evaluations_Analysis.ipynb`: Jupyter notebook presenting the IR and LLM evaluation results and the analysis of their relationship.

You can run this notebook on Google Colab or in the Jupyter/Scipy-Notebook Docker container using:

```sh
docker run --network host --rm --gpus all -d -p 7880:8888 --shm-size=1g -e NVIDIA_VISIBLE_DEVICES=all --user root -e NB_UID=`id -u` -e NB_GID=`id -g`  -v "$PWD":/home/jovyan/scripts jupyter/scipy-notebook
```

or you can run it in any other Jupyter notebook environment.



## Illustrations

### Illustration of the study conducted to evaluate Retrieval-Augmented Generation (RAG) systems

![Illustration of Study](https://github.com/user-attachments/assets/eeb31faf-cf0d-440d-b486-5d3955399632)

### Illustration of our evaluation framework

It begins by assessing the Information Retrieval (IR) system performance and directly estimates the overall RAG performance from it, eliminating the need for expensive response generation and evaluation.

![Evaluation Framework](https://github.com/user-attachments/assets/5e99bdcc-5e0b-4186-9078-bea1c1e3f2bb)

