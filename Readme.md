[![Dataset: Dolly-15k](https://img.shields.io/badge/Dataset-Dolly--15k-orange)](https://huggingface.co/datasets/databricks/databricks-dolly-15k)
[![Model: Llama-3.2-1B](https://img.shields.io/badge/Model-Llama--3.2--1B-blue)](https://huggingface.co/meta-llama)
[![Hugging Face: Adapter](https://img.shields.io/badge/HF-Adapter--QLoRA-yellow)](https://huggingface.co/aadhil2k4/llama3.2-1b-dolly15k-qlora-adapter)
[![Hugging Face: Merged Model](https://img.shields.io/badge/HF-Merged--Model-blue)](https://huggingface.co/aadhil2k4/llama3.2-1b-dolly15k-qlora)

# 🦙 Llama-3.2-1B QLoRA Fine-tuning (PEFT + TRL)

Instruction-tuning **Llama-3.2-1B** on **Databricks Dolly-15k** using **QLoRA** and **PEFT**, converting the base model into an instruction-following model.  
This notebook demonstrates low-VRAM fine-tuning with 4-bit quantization and LoRA adapters.

---


## 🧩 Model Checkpoints

| Type | Hugging Face Link | Description |
|------|--------------------|--------------|
| 🧠 **Adapter-only (QLoRA)** | [aadhil2k4/llama3.2-1b-dolly15k-qlora-adapter](https://huggingface.co/aadhil2k4/llama3.2-1b-dolly15k-qlora-adapter) | LoRA adapter weights for PEFT loading. Lightweight and ideal for continuing fine-tuning. |
| 🦙 **Merged model (Full fine-tuned)** | [aadhil2k4/llama3.2-1b-dolly15k-qlora](https://huggingface.co/aadhil2k4/llama3.2-1b-dolly15k-qlora) | Base + adapter merged. Ready for direct inference without PEFT. |

> 💡 The adapter version is much smaller (few hundred MB), while the merged model is full-sized (~2–3 GB). Use the adapter for further fine-tuning or lightweight deployment.

## 🚀 Overview

Large models like Llama are powerful but heavy to train.  
**QLoRA (Quantized LoRA)** lets us fine-tune models efficiently on smaller GPUs by:
- Loading models in **4-bit** quantization (NF4)
- Training **LoRA adapters** on selected layers
- Using **PEFT** + **TRL** for streamlined fine-tuning

---

## 📘 Notebook

All code is inside: notebooks/finetune_qlora_llama32_1b_dolly15k.ipynb

This notebook walks through the complete fine-tuning workflow:
1. **Dataset loading** — Uses `databricks/databricks-dolly-15k` from Hugging Face Datasets.  
2. **Preprocessing** — Formats samples into `Instruction → Response` pairs (and includes context when available).  
3. **Model preparation** — Loads `meta-llama/Llama-3.2-1B` in 4-bit (NF4) mode using `bitsandbytes`.  
4. **PEFT configuration** — Applies LoRA adapters (`r=8, alpha=16, dropout=0.05`) to attention and MLP layers.  
5. **Training** — Runs supervised fine-tuning with TRL’s `SFTTrainer` for one epoch.  
6. **Saving & merging** — Exports both the adapter and a merged version of the model.  
7. **Inference** — Demonstrates generation examples using base, adapter-loaded, and merged models.

> 💡 The notebook is fully self-contained — you can open it in Jupyter or Google Colab and run it cell by cell.
> Small Note: During this work, it was ran on Kaggle.

---

## 🧠 Training Configuration

| Parameter | Value |
|------------|--------|
| **Base model** | `meta-llama/Llama-3.2-1B` |
| **Dataset** | `databricks/databricks-dolly-15k` |
| **LoRA rank (r)** | 8 |
| **LoRA alpha** | 16 |
| **LoRA dropout** | 0.05 |
| **Quantization** | 4-bit NF4 |
| **Batch size** | 2 |
| **Gradient accumulation** | 8 |
| **Learning rate** | 2e-4 |
| **Epochs** | 1 |

## 💬 Inference Example

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

base_model = "meta-llama/Llama-3.2-1B"
MODEL = "Kenjiii/llama3.2-1b-dolly15k-qlora"

tok = AutoTokenizer.from_pretrained(MODEL, legacy=True)
tok.pad_token = tok.eos_token


base_model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

prompt = "Explain what a GPU is, then give an example of how it’s used in AI."
text   = f"Instruction:\n{prompt}\n\nResponse:\n"
inputs = tok([text], return_tensors="pt").to(model.device)
out    = base_model.generate(
    **inputs, 
    max_new_tokens=150
)
print(tok.decode(out[0], skip_special_tokens=True))

prompt = "Explain what a GPU is, then give an example of how it’s used in AI."
text   = f"Instruction:\n{prompt}\n\nResponse:\n"
inputs = tok(text, return_tensors="pt").to(model.device)
outputs_adapter = model.generate(**inputs, max_new_tokens=150)
print(tok.decode(outputs_adapter[0], skip_special_tokens=True))
```

## 🧩 Sample Generations

Below are examples comparing **base model** vs **instruction-tuned model (QLoRA adapter)** on the same prompt.

### 🧠 Prompt
> Explain what a GPU is, then give an example of how it’s used in AI.

---

**Base Model (`meta-llama/Llama-3.2-1B`):**
> A GPU is a graphics processing unit. It’s a special type of processor designed to accelerate graphics processing. GPUs are used in AI because they can process large amounts of data quickly, which makes them ideal for tasks like training neural networks and performing image recognition.  
>  
> In this course, we’ll explore how GPUs are used to accelerate machine learning…

---

**Fine-Tuned Model (QLoRA on Dolly-15k):**
> A GPU, or graphics processing unit, is a specialized chip used to accelerate tasks such as training and inference of machine learning models.  
> For example, a GPU may speed up the training of a neural network that predicts customer behavior or analyzes images in real time.

---

> 🟢 *The fine-tuned model produces shorter, more direct, and instruction-focused responses compared to the base model.*

---

---

## 🔮 Future Work

This project focused on **Supervised Fine-Tuning (SFT)** using the Dolly-15k dataset to teach the base model to follow human instructions.  
The next logical step is to continue toward full **alignment** through **Reinforcement Learning from Human Feedback (RLHF)**.

Planned directions include:
- 🧠 **Reward Model Training** — Train a reward model to score generated responses based on quality, helpfulness, and safety.  
- 🎯 **Preference Optimization** — Use algorithms such as **PPO (Proximal Policy Optimization)** or **DPO (Direct Preference Optimization)** to further align model outputs with human preferences.  
- ⚙️ **Evaluation Framework** — Develop metrics to quantitatively measure instruction-following quality and response coherence.  
- 🌐 **Extended Datasets** — Experiment with larger or more diverse instruction datasets for better generalization.

> 💡 The goal of future work is to move from *instruction-following* to *preference-aligned* models, completing the SFT → RM → PPO pipeline.

