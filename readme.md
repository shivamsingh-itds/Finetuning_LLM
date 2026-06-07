# 🧠 Fine-Tuning, LoRA, and QLoRA 

## 🔍 What is Fine-Tuning?

Fine-Tuning is the process of:

> Taking a pre-trained Large Language Model (LLM) and training it further on a specific dataset to make it perform better on a particular task.

Instead of training a model from scratch, we start with an already trained model and adapt it to our use case.

Examples:
- Chatbots
- Customer support assistants
- Medical question answering
- Code generation

---

# 🎯 Why Do We Need Fine-Tuning?

Pre-trained LLMs:
- Have general knowledge
- May not know domain-specific information
- May not follow desired response styles

Fine-tuning helps:
- Improve task-specific performance
- Learn custom behavior
- Adapt to domain-specific data

---

# 📌 Traditional Fine-Tuning

In traditional fine-tuning:

- All model parameters are updated
- Entire model is retrained

Example:

```text
Pretrained Model
        ↓
Update All Weights
        ↓
Fine-Tuned Model
```

---

# ⚠️ Problems with Traditional Fine-Tuning

For modern LLMs:

- Billions of parameters
- Requires huge GPU memory
- Expensive training
- Large storage requirements

Example:

```text
Llama 7B → Billions of parameters
```

Fine-tuning all parameters becomes costly.

---

# 🧠 What is PEFT?

PEFT stands for:

> Parameter Efficient Fine-Tuning

Instead of updating all weights:

- Only a small number of parameters are trained

Benefits:
- Less memory
- Faster training
- Lower cost

Popular PEFT methods:

- LoRA
- QLoRA
- Prefix Tuning
- Prompt Tuning

---

# 🧠 What is LoRA?

LoRA stands for:

> Low-Rank Adaptation

It is a PEFT technique used to fine-tune LLMs efficiently.

Instead of updating the original model weights:

- Original weights remain frozen
- Small trainable matrices are added

---

# 🎯 Main Idea of LoRA

Instead of modifying:

\[
W
\]

LoRA learns:

\[
W + \Delta W
\]

Where:

\[
\Delta W = A \times B
\]

A and B are small low-rank matrices.

---

# 📊 LoRA Architecture

```text
Input
   ↓
Frozen Pretrained Weights
   ↓
LoRA Adapters
   ↓
Output
```

Only LoRA adapters are trained.

---

# 📌 Advantages of LoRA

- Very memory efficient
- Faster training
- Small model checkpoints
- High performance
- Easy deployment

---

# ⚙️ LoRA Hyperparameters

## Rank (r)

Controls adapter size.

Example:

```python
r = 8
```

Higher rank:
- More trainable parameters
- Better learning
- More memory usage

---

## Alpha

Scaling factor.

Example:

```python
lora_alpha = 16
```

Controls LoRA update strength.

---

## Dropout

Regularization for adapters.

Example:

```python
lora_dropout = 0.05
```

---

# 🧠 What is QLoRA?

QLoRA stands for:

> Quantized Low-Rank Adaptation

QLoRA combines:

- Quantization
- LoRA

to further reduce memory usage.

---

# 🎯 Main Idea of QLoRA

Instead of loading the model in:

```text
16-bit (FP16)
```

QLoRA loads the model in:

```text
4-bit Quantization
```

and then applies LoRA adapters.

---

# 📊 QLoRA Workflow

```text
Pretrained Model
        ↓
4-bit Quantization
        ↓
LoRA Adapters
        ↓
Fine-Tuning
```

---

# 📌 Why QLoRA is Powerful?

Benefits:

- Extremely low memory usage
- Can fine-tune large models on a single GPU
- Similar performance to full fine-tuning

Example:

```text
Llama 7B
Mistral 7B
Gemma
```

can be fine-tuned on consumer GPUs.

---

# 🔄 Fine-Tuning vs LoRA vs QLoRA

| Feature | Full Fine-Tuning | LoRA | QLoRA |
|----------|----------------|------|--------|
| Update All Parameters | Yes | No | No |
| Memory Usage | High | Low | Very Low |
| Training Cost | High | Low | Very Low |
| Speed | Slow | Faster | Fastest |
| Storage | Large | Small | Very Small |

---

# ⚙️ LoRA Example using PEFT

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05
)
```

---

# ⚙️ QLoRA Example

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True
)
```

This loads the model in 4-bit precision.

---

# 📌 When to Use Fine-Tuning?

Use Full Fine-Tuning when:

- Huge GPU resources available
- Maximum performance needed

---

# 📌 When to Use LoRA?

Use LoRA when:

- Limited GPU memory
- Faster training required
- Efficient deployment needed

---

# 📌 When to Use QLoRA?

Use QLoRA when:

- Consumer GPU available
- Very large models need tuning
- Memory optimization is important

---

# ⭐ Advantages of LoRA

- Memory efficient
- Faster training
- Small checkpoints
- Easy to share adapters

---

# ⭐ Advantages of QLoRA

- Extremely memory efficient
- Supports very large LLMs
- Similar performance to full fine-tuning
- Lower hardware requirements

---

# ⚠️ Limitations

## LoRA

- Slightly lower flexibility than full fine-tuning

## QLoRA

- Quantization may introduce minor information loss
- Slightly more complex setup

---

# Key Takeaway

> Fine-Tuning adapts a pretrained LLM to a specific task. LoRA achieves this efficiently by training small adapter matrices instead of all model parameters, while QLoRA further reduces memory usage by combining LoRA with 4-bit quantization, enabling large language models to be fine-tuned on affordable hardware.