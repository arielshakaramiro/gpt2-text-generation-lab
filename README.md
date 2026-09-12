# GPT-2 Text Generation Lab

Two hands-on case studies in NLP text generation using GPT-2: training a language model **from scratch** for Python code completion and deploying it as an API, plus **fine-tuning** a pretrained Indonesian GPT-2 on a custom dataset. Built with Hugging Face `transformers`, `datasets`, and FastAPI.

## What's in here

| # | Notebook | Case study | Approach |
|---|---|---|---|
| 1 | [`01_code_generation_from_scratch.ipynb`](notebooks/01_code_generation_from_scratch.ipynb) | Python code completion | Train a GPT-2 architecture **from random weights** on a pre-filtered subset of the CodeParrot dataset (the notebook's own filtering step is included but skipped in this run — see Setup) |
| 2 | [`02_deploy_text_generation_api.ipynb`](notebooks/02_deploy_text_generation_api.ipynb) | Serving | Wrap the model from notebook 1 in a **FastAPI** endpoint, exposed publicly via **ngrok** |
| 3 | [`03_finetune_gpt2_indonesia.ipynb`](notebooks/03_finetune_gpt2_indonesia.ipynb) | Indonesian text generation | Explore pretrained Indonesian BERT/RoBERTa/GPT-2, then **fine-tune** `cahya/gpt2-small-indonesian-522M` on a custom dataset |

Full run outputs (training logs, generation samples) are preserved in the notebooks. See [`docs/RESULTS.md`](docs/RESULTS.md) for the summarized, annotated findings.

## Why two different approaches?

Training from scratch (notebook 1) and fine-tuning a pretrained model (notebook 3) are genuinely different strategies with different trade-offs. Both were run here on a small, Colab-friendly compute budget, which happens to make the contrast clear: notebook 1 shows what from-scratch training looks like before it has had enough steps to produce coherent output, while notebook 3 shows how quickly fine-tuning converges — and the overfitting risk that comes with that speed on a tiny dataset. See `docs/RESULTS.md` for the concrete numbers and an honest read of what each result does and doesn't show.

## Setup

All notebooks are designed to run on **Google Colab** with a GPU runtime.

1. Open a notebook in Colab.
2. Run cells top to bottom. Notebook 1 has one clearly marked **optional** cell (raw dataset filtering, ~50GB) — skip it and use the pre-filtered dataset in the next cell instead.
3. **Notebook 1** saves the trained model to Google Drive at `/content/drive/MyDrive/text-generation-project/code-generator-model` — you'll be prompted to authorize Drive access.
4. **Notebook 2** needs an [ngrok](https://ngrok.com/) auth token stored as a Colab Secret named `NGROK_AUTHTOKEN` (key icon 🔑 in the left sidebar) — never hardcode it in a cell.
5. **Notebook 3** is self-contained (no Drive dependency); it trains and saves locally within the Colab session.

## Tech stack

`transformers` · `datasets` · `accelerate` · `evaluate` · PyTorch · FastAPI · `uvicorn` · `pyngrok`

## Repo structure

```
gpt2-text-generation-lab/
├── README.md
├── LICENSE
├── .gitignore
├── notebooks/
│   ├── 01_code_generation_from_scratch.ipynb
│   ├── 02_deploy_text_generation_api.ipynb
│   └── 03_finetune_gpt2_indonesia.ipynb
└── docs/
    └── RESULTS.md
```

## Acknowledgements

- Notebook 1's from-scratch training approach follows the [Hugging Face NLP Course](https://huggingface.co/learn/nlp-course), Chapter 7 ("Training a causal language model from scratch") — released under the Apache 2.0 license.
- The CodeParrot dataset is from [`transformersbook/codeparrot-ds`](https://huggingface.co/datasets/transformersbook/codeparrot), built for the *Natural Language Processing with Transformers* book.
- The Indonesian models used in notebook 3 (`cahya/gpt2-small-indonesian-522M`, `cahya/bert-base-indonesian-522M`, `cahya/roberta-base-indonesian-522M`) are community models by [cahya](https://huggingface.co/cahya) on the Hugging Face Hub.

## License

This repository's own code and documentation are licensed under the [MIT License](LICENSE). This does not extend to third-party datasets or pretrained model weights used at runtime (see Acknowledgements) — those remain under their own respective licenses and are not redistributed in this repo.
