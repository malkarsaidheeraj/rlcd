# RLCD: PyTorch learning examples and a toy pipeline

This repository records small PyTorch exercises that lead to a **toy RLCD-style demonstration**. The scripts run on Apple Silicon through PyTorch MPS when available and fall back to CPU elsewhere.

## Run

Use Python 3 with PyTorch installed. No other packages are needed. From this folder:

```bash
python3 train_linear.py
python3 two_layer_model.py
python3 text_inputs.py
python3 next_token.py
python3 rlcd_toy.py
```

| File | Concept |
| --- | --- |
| `train_linear.py` | Gradients and SGD fit `y = 2x + 1`. |
| `two_layer_model.py` | A two-layer ReLU network fits `y = abs(x)` and checks new inputs. |
| `text_inputs.py` | Word IDs, embeddings, simple attention weights, and a causal mask. |
| `next_token.py` | A one-token-context model predicts the next ID in `the cat sat`. |
| `rlcd_toy.py` | Contrasting answer pairs, a learned preference score, and a reward-guided policy update. |

## How the RLCD-style example works

For each of three questions, a positive instruction selects an accurate answer and a negative instruction selects an inaccurate one. These pairs train a reward model to score the preferred answer higher. A separate policy starts with equal probability for the two choices and is optimized for the reward model's scores, with a KL penalty that discourages moving too far from the starting policy. The script prints training-pair ranking accuracy and the preferred-answer probability before and after the update.

The answers and instruction effects are scripted. The reward model is a lookup table over the six answers, and the policy uses exact expected reward over two choices. Its ranking accuracy is measured on its **training pairs**, so it does not demonstrate generalization. This is a small illustration of the pipeline, not a language-model RLCD reproduction: it does not generate open-ended text or use PPO. The [RLCD paper](https://proceedings.iclr.cc/paper_files/paper/2024/hash/5bd09a559a8c8e230697107b0f353d39-Abstract-Conference.html) and [authors' code](https://github.com/facebookresearch/RLCD) describe the full method.
