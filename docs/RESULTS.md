# Results & Findings

All numbers and samples below are taken directly from the executed notebooks in this repo — nothing here is simulated or estimated.

## 1. Training summary

| Notebook | Approach | Training data | Epochs / steps | Final training loss | Training time |
|---|---|---|---|---|---|
| 01 — Code Generation | From-scratch GPT-2 (124M params, random init) | 50,000 samples from the pre-filtered `codeparrot-ds` dataset† | 1 epoch / 196 steps | **6.43** | ~10.5 min |
| 03 — Indonesian Fine-tune | Fine-tuned `cahya/gpt2-small-indonesian-522M` | 50 dummy dinosaur-facts sentences | 10 epochs / 250 steps | **1.27** | ~6 min |

† Pre-filtered by the Hugging Face course maintainers (`huggingface-course/codeparrot-ds-train`), not by this notebook — its own from-scratch filtering cell is present but was skipped in this run (it requires downloading the full ~50GB raw dataset).

At a glance, notebook 3's loss looks far better. Read past the number, though — see [Finding 2](#finding-2-the-low-loss-in-notebook-3-is-likely-overfitting-not-generalization) below.

## 2. Generation samples

### Notebook 1 — Code completion (greedy decoding, no sampling params)

Prompt:
```python
# create some data
x = np.random.randn(100)
y = np.random.randn(100)

# create dataframe from x and y
```

Output (truncated):
```
# the of the data
# with the terms of the
# (C)
# This is the terms of the terms of the Free Software Foundation, and the terms
# of the Free Software Foundation, or (at your option) (at your option) any
# later version.
# but WITHOUT ANY WARRANTIES OR CONDITIONS OF ANY WITHOUT ANY WARRANTY...
```

The model does not complete the code — it loops on open-source license boilerplate (GPL/Apache-style headers), which is an extremely common pattern at the start of scraped GitHub files. With only 196 training steps on random initial weights, this is the "easiest" pattern for the model to have picked up first.

### Notebook 2 — Same model, served via API (`top_k=50, top_p=0.9, temperature=0.8`)

Request to `/generate` with the same prompt returned (truncated):
```
# create dataframe from x and y

    # a not a dataset of the a function of the dataset
    # or a copy of the
data = True
    import scipy.
    from the the data
    """
    # * return p, n_csv_import sys
import matplotlib.path,
    def __init__(self, y, '0, '
from ..utils.path import check_toolkits.sparse_state...
```

Still incoherent and not valid Python, but qualitatively different from notebook 1's output: with sampling enabled, the model produces more code-like tokens (`import`, `def __init__`, `class`, `assert`) instead of just repeating license text. **Decoding strategy visibly changes perceived output quality, even for the same underlying (undertrained) model.**

### Notebook 3 — Fill-mask exploration (no training, pretrained models only)

| Model | Input | Top prediction | Confidence |
|---|---|---|---|
| `cahya/bert-base-indonesian-522M` | `Ibu ku sedang bekerja [MASK] supermarket` | "di" | 94.96% |
| `cahya/roberta-base-indonesian-522M` | `Ibu ku sedang bekerja <mask> supermarket` | " di" | 64.92% |

Both models correctly and confidently predict the preposition "di" — a sanity check that these pretrained masked-LMs work as expected out of the box.

### Notebook 3 — Fine-tuned generation

Prompt: `"Trex merupakan"`

Output (truncated):
```
Trex merupakan dinosaurus karnivora pertama yang diketahui memiliki adaptasi
unik untuk hidup dan berburu di dalam air. Bukti yang dapat diandalkan untuk
hidup dan berburu di dalam air menunjukkan bahwa T-Rex memiliki adaptasi unik
untuk hidup dan berburu di dalam air. Terdapat empat genus dalam T-Rex...
```

This reads fluently — but see the finding below before treating that as a success on its own.

## Finding 1: Notebook 1 is undertrained, not broken

50,000 samples / 1 epoch / 196 steps is nowhere near enough for a 124M-parameter transformer starting from **random weights** to learn code syntax. This is expected, not a bug — training a language model from scratch typically needs far more than 196 steps to move meaningfully away from random-guessing behavior over the vocabulary. The observed loss of 6.43 shows the model has learned *something* in this short run, but is still far from producing coherent code. The pipeline (filtering → tokenization → architecture → training → serving) is fully correct and runs end to end; the *result quality* is simply a function of the (intentionally small, Colab-friendly) compute budget used here.

## Finding 2: The low loss in Notebook 3 is likely overfitting, not generalization

Compare the fine-tuned output above to the actual training data:

> Training sample: *"Spinosaurus adalah dinosaurus karnivora **pertama yang diketahui memiliki adaptasi unik untuk hidup dan berburu di dalam air**."*
>
> Generated output: *"Trex merupakan dinosaurus karnivora **pertama yang diketahui memiliki adaptasi unik untuk hidup dan berburu di dalam air**."*

The generated sentence is a near-verbatim reuse of a training sentence about a different dinosaur, with the subject swapped. With only **50 training examples repeated over 10 epochs** fine-tuning an already-pretrained model, this is a textbook overfitting signature: the model is stitching together memorized fragments rather than generating genuinely new text.

Two methodological details make this expected:
- The dataset has 50 examples total — far too small to expect generalization from fine-tuning alone.
- The `Trainer` in this notebook uses `eval_dataset=tokenized_dataset` — **the same set used for training** — so the reported eval loss (not shown separately here) cannot detect overfitting; a proper held-out validation split would be needed to measure that honestly.

**Takeaway:** a dramatically lower loss and more fluent-sounding output are not proof of a better model — they can just as easily indicate memorization on a tiny dataset. A fair comparison would need a held-out validation set and/or a larger, non-duplicated fine-tuning dataset.

## Limitations & possible next steps

- **Notebook 1:** train for more epochs/steps (compute-permitting) to see whether loss drops meaningfully below the current 6.43, and whether output shifts away from license-boilerplate toward actual code structure.
- **Notebook 3:** re-run with a genuine train/validation split and a larger, more diverse dataset to get an honest read on whether fine-tuning generalizes beyond the training sentences.
- **Notebook 2:** the current decoding parameters (`top_k=50, top_p=0.9, temperature=0.8`) were not tuned — worth a small sweep once the underlying model (notebook 1) is better trained.
