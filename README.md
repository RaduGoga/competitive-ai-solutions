# AI Competition Solutions
My solutions to **122 tasks** (165 notebooks and scripts) from Romanian AI olympiads and competitive machine-learning platforms, national olympiad rounds (ONIA / ROAI), practice contests, and training problems.
These competitions follow the format of the [IOAI](https://ioai-official.org/) (International Olympiad in Artificial Intelligence): Kaggle-style tasks, solved under a hard time limit, without internet access and without LLM assistance. 
Every solution here is my own work, written under those conditions or while practicing for them (practice solutions may include short copied snippets).

The two platforms hosting these contests:
-  **MLCompete** - [platform.olimpiada-ai.ro](https://platform.olimpiada-ai.ro/en/problems)

-  **Nitro AI Judge** - [judge.nitro-ai.org](https://judge.nitro-ai.org/competitions)

> **[Full index of every solved task + approach](SOLUTIONS.md)**

## Repository structure
```
Platforms/
  MLCompete/
    <standalone task>/
      sol.ipynb
    _Competitions/
      <contest>/
        <task>/
          sol.ipynb
  Nitro AI Judge/
    <contest>/
      <grade tier or round>/
        <task>/
          sol.ipynb
          sol_2.ipynb
```
Task folders have their original Romanian names as they appear on the platform. Where a task has more solutions side by side (`sol.ipynb`, `sol_resnet.ipynb`, `sol_lstm.ipynb`, etc.), those are different approaches to the same problem, kept for comparison.

## A note on data
**Datasets are not included in this repository.** The full working tree obviously exceeds GitHub's 100 MB hard limit, and none of the datasets are mine to redistribute.
To run a solution:
1. Download the task's dataset from the platform page (MLCompete or Nitro AI Judge).
2. Unpack it **into the task's folder**, next to the notebook.
3. Start Jupyter with the working directory set to that folder, then run top to bottom.

## Technologies
The packages used in the problems are included in the **requirements.txt**.

The problems span classical tabular ML (regression, classification), computer vision (CNNs, transfer learning with ResNet/VGG, segmentation, classical methods), NLP (TF-IDF, Word2Vec embeddings, LSTMs, seq2seq, transformers), audio classification, and some generative work.
Install the common dependencies with:

```bash
pip  install  -r  requirements.txt
```

## Solved Competitions
A more detailed catalogue with approaches can be found in **[SOLUTIONS.md](SOLUTIONS.md)**.

| Platform | Contest | Tasks |
| --- | --- | --- |
| MLCompete | Standalone practice tasks | 61 |
| MLCompete | ONIA 2026 | 15 |
| Nitro AI Judge | ROAI 2026 | 8 |
| Nitro AI Judge | ONIA 2025 | 5 |
| Nitro AI Judge | AIIS 2026 | 3 |
| Nitro AI Judge | RISE 2026 | 4 |
| Nitro AI Judge | Many other small contests | 26 |

## License
Written by **Ilie-Goga Radu**. Licensed under the [MIT License](LICENSE).
The problem statements and datasets belong to their respective platforms and are not part of this repository — only my solutions are.