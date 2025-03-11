## Introduction
This is the official implementation of ICLR 2025 paper "Can In-context Learning Really Generalize to Out-of-distribution Tasks?"

[[paper]](https://openreview.net/pdf?id=INe4otjryz)

## Setup
```bash
conda env create -f environment.yml
conda activate in-context-learning
```

## Directly reproduce the experiments
### Reproduce Fig. 1,3,4,5,7 (GPT-2 experiments)
Please see eval_multi_curve.ipynb.

### Reproduce Fig. 2,6,8,7 (LLM experiments)
Coming soon.

## Train and evaluate your own model
### Training

```bash
cd ./src
conda activate in-context-learning
python train.py --config conf/linear_regression.yaml
```

The trained model weights will be saved to `./results/trained_ckpt_and_eval_results/`. Note that we didn't upload the trained checkpoints in this direction.

### Evaluation

```bash
python -m eval_ood_task --pretrain_path ./results/trained_ckpt_and_eval_results/linear_regression/[random-id-generated-by-your-system] \
    --ood_task quadratic_regression --device cuda:5 --n_context_test 101
```

The raw evaluation results are saved in `./results/trained_ckpt_and_eval_results/` for displaying in `./src/eval_multi_curve.py`. Each folder corresponds to a pretraining task, and the evaluation results of different evaluation tasks are together recorded in `metrics.json` under each folder.

## Other details
We also maintain the raw evaluation results of some other function classes in `./results/trained_ckpt_and_eval_results/`. We didn't include these results in our paper to avoid making the paper too lengthy, nevertheless, we find some of them insteresting. You can simply identify the pretraining task by observing the path name to get the corresponding result and display it by adding it in `./src/eval_multi_curve.py`.