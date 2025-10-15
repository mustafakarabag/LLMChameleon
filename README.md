# Do LLMs Strategically Reveal, Conceal, and Infer Information? A Theoretical and Empirical Analysis in The Chameleon Game

Repository for the paper ["Do LLMs Strategically Reveal, Conceal, and Infer Information? A Theoretical and Empirical Analysis in The Chameleon Game"](http://arxiv.org/abs/2501.19398). This repository contains the gameplay code to make different LLMs play The Chameleon, and utilities for analysis.


## Environment setup
Copy `.env.example` to `.env` and set all the environment variables. For variables pointing to a directory, create the directory if it does not exist.

Set up an environment from the `environment.yaml` file and activate it ([Miniconda](https://docs.anaconda.com/free/miniconda/index.html)):
```bash
conda env create -f environment.yaml
conda activate llmg
```
With the environment activated, install the local `llmg` package:
```bash
pip install -e .
```


## Running the code
If you don't have the original Chameleon game cards, or want to create new ones, use the notebook `new_categories.ipynb` in the `llmg/chameleon` folder. This notebook prompts GPT-4o to generate new categories and secret words, and saves them in `new_chameleon_cards_<timestamp>.pkl`. See the comments in the notebook for details. *(The Chameleon is a commercial board game which you can buy at [bigpotato.com/products/the-chameleon](https://bigpotato.com/products/the-chameleon))*

The notebook `data_collection.ipynb` in the `llmg/chameleon` folder contains the code to run the experiments and collect data. The results will be saved in the directory specified by the `DATA_DIR` environment variable.

The notebook `analysis.ipynb` in the `llmg/chameleon` folder contains the code to analyze the collected data (e.g., compute non-chameleon win ratios), including evaluating the response words with an LLM evaluator.

The notebook `strategy_steering.ipynb` in the `llmg/chameleon` folder contains the code to compute and apply strategy steering vectors to the LLMs (only for PCA plotting, steered gameplay is done in `data_collection.ipynb`).

All the other files are utilities and modules used in the notebooks.
