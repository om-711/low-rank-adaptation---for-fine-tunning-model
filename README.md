# LoRA From Scratch in PyTorch

## Overview

This project demonstrates how **LoRA (Low-Rank Adaptation)** works internally using a simple neural network built completely from scratch in PyTorch.

Instead of directly using large transformer models like GPT or LLaMA, this implementation first builds intuition using a small feedforward neural network.

The project compares:

1. Full Fine-Tuning
2. LoRA Fine-Tuning

and shows:

* how LoRA reduces trainable parameters,
* how pretrained weights are frozen,
* how low-rank matrices are added,
* and how LoRA adapts a pretrained model efficiently.

---

# What is LoRA?

LoRA stands for:

```text
Low-Rank Adaptation
```

Normally during fine-tuning, neural network weights are updated directly.

Suppose a layer has weight matrix:

```math
W
```

Full fine-tuning learns:

```math
W' = W + ΔW
```

where:

* `W` = pretrained weights
* `ΔW` = learned update

---

## Core Idea of LoRA

LoRA assumes:

```text
The update matrix ΔW does not need full complexity.
```

Instead of learning a huge matrix directly, LoRA approximates the update using two smaller matrices:

```math
ΔW = BA
```

where:

* `A` = low-rank matrix
* `B` = low-rank matrix
* `rank << original dimensions`

Thus the layer becomes:

```math
Wx + BAx
```

where:

* `Wx` = frozen pretrained model output
* `BAx` = trainable LoRA update

---

# Why LoRA?

Full fine-tuning large models is expensive because:

* all weights need gradients,
* optimizer states become huge,
* memory usage increases massively.

LoRA solves this by:

* freezing original weights,
* training only tiny low-rank matrices.

This drastically reduces:

* GPU memory usage,
* trainable parameters,
* storage requirements.

---

# Project Workflow

The project follows this pipeline:

```text
Train Normal Model
        ↓
Save Pretrained Weights
        ↓
Create New Task
        ↓
Full Fine-Tuning
        ↓
LoRA Fine-Tuning
        ↓
Compare Results
```

---

# Step 1 — Creating the Original Model

A simple feedforward neural network is created:

```python
class SimpleModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(1, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 1)
```

Architecture:

```text
1 → 16 → 32 → 1
```

This model acts as the:

```text
Pretrained Foundation Model
```

---

# Step 2 — Pretraining

The model is first trained on:

```math
y = 2x + 1
```

Example:

```python
x_train = torch.randn(1000,1)

y_train = 2*x_train + 1
```

This stage simulates:

```text
Pretraining
```

where the model learns general knowledge.

---

# Step 3 — Saving the Pretrained Model

```python
torch.save(model.state_dict(), 'pretrained.pth')
```

This stores the learned weights.

---

# Step 4 — Creating a New Task

Now the task is changed:

```math
y = 3x - 2
```

Example:

```python
x_new = torch.randn(1000, 1)

y_new = 3 * x_new - 2
```

This simulates:

```text
Domain Adaptation / Downstream Fine-Tuning
```

---

# Step 5 — Full Fine-Tuning

In full fine-tuning:

```text
ALL model parameters are updated.
```

Code:

```python
optimizer_full = torch.optim.Adam(
    full_model.parameters(),
    lr=0.01
)
```

This gives maximum flexibility but requires training all weights.

---

# Step 6 — Implementing LoRA

## LoRA Layer

```python
class LoRA(nn.Module):

    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
```

This module creates the low-rank update matrices.

---

## Matrix A

```python
self.A = nn.Parameter(
    torch.randn(in_dim, rank) * 0.01
)
```

### Why multiply by `0.01`?

Random initialization with large values can create unstable updates.

LoRA wants:

```text
Very small updates initially
```

so the pretrained model behavior is preserved.

---

## Matrix B

```python
self.B = nn.Parameter(
    torch.zeros(rank, out_dim)
)
```

### Why initialize B with zeros?

Initially:

```math
BA = 0
```

Thus:

```math
Wx + BAx = Wx
```

Meaning:

```text
The model initially behaves EXACTLY like the pretrained model.
```

This is one of the most important stability tricks in LoRA.

---

# LoRA Forward Pass

```python
x = (self.alpha / self.rank ) * (x @ self.A @ self.B)
```

This computes:

```math
\frac{\alpha}{r}(xAB)
```

where:

* `alpha` controls update strength,
* `rank` controls low-rank capacity.

---

# Step 7 — Creating the LoRA Model

The original linear layer and LoRA update are combined:

```python
x = self.fc1(x) + self.lora1(x)
```

This becomes:

```math
Wx + BAx
```

which is the actual LoRA equation.

---

# Why Freeze Original Layers?

```python
for param in self.fc1.parameters():
    param.requires_grad = False
```

This freezes pretrained weights.

LoRA philosophy is:

```text
Do NOT change pretrained knowledge.
Only learn small task-specific corrections.
```

---

# Why Pretrained Weights Must Be Copied

This is one of the most important concepts.

When:

```python
pretrained.load_state_dict(...)
```

is called, weights are loaded ONLY into:

```python
pretrained
```

NOT into:

```python
lora_model
```

because they are different objects in memory.

Therefore pretrained weights must be copied manually:

```python
lora_model.fc1.weight.data.copy_(
    pretrained.fc1.weight.data
)
```

Without copying:

```text
LoRA would train on random frozen weights.
```

which defeats the purpose of LoRA.

---

# Step 8 — Training LoRA

Only LoRA parameters are optimized:

```python
optimizer_lora = torch.optim.Adam(

    filter(
        lambda p: p.requires_grad,
        lora_model.parameters()
    ),

    lr=0.01
)
```

This ensures:

```text
ONLY LoRA matrices receive gradients.
```

---

# Parameter Comparison

## Full Fine-Tuning

All parameters are trainable.

```python
trainable_params_full = sum(
    p.numel()
    for p in full_model.parameters()
    if p.requires_grad
)
```

---

## LoRA Fine-Tuning

Only LoRA matrices are trainable.

```python
trainable_params = sum(
    p.numel()
    for p in lora_model.parameters()
    if p.requires_grad
)
```

This demonstrates the parameter efficiency of LoRA.

---

# Experimental Observation

Example result:

```text
Expected Value:
4

Full FT Prediction:
3.988

LoRA Prediction:
3.026
```

---

# Why Did LoRA Perform Worse?

The experiment used:

```text
rank = 2
alpha = 0.2
```

which gives LoRA very limited adaptation capacity.

This demonstrates an important concept:

```text
LoRA trades adaptation power for parameter efficiency.
```

Increasing:

* rank,
* alpha,

usually improves LoRA performance.

---

# Important Insights Learned

This project demonstrates:

* how LoRA works mathematically,
* how pretrained weights are frozen,
* how low-rank updates are learned,
* why LoRA saves parameters,
* why LoRA needs pretrained weights,
* and how LoRA compares with full fine-tuning.

---

# Final Core Idea

LoRA replaces:

```math
W' = W + ΔW
```

with:

```math
W' = W + BA
```

where:

* `W` = frozen pretrained weights,
* `BA` = small trainable low-rank update.

This allows efficient fine-tuning with drastically fewer trainable parameters.
