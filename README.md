
# **Training Gemma to Show Its Work with GRPO and Tunix**

This repository contains the complete training pipeline for fine-tuning the **Gemma 3 1B-IT** open-weight language model to produce **structured reasoning traces** using **Google Tunix** and **Group Relative Policy Optimization (GRPO)** on Kaggle TPUs.

The goal of this project is to teach a language model not only to give correct answers, but also to explicitly show how it arrived at those answers.

---

## 🚀 Project Overview

Most large language models generate answers without reliably exposing their reasoning process. This project uses **reinforcement learning–based post-training** to encourage the model to output step-by-step reasoning inside special tags:

```
<reasoning> ... </reasoning>
<answer> ... </answer>
```

This makes the model more transparent, interpretable, and useful for tasks like math problem solving and decision support.

---

## 🧠 Model & Training

* **Base Model**: Gemma 3 (1B-IT)
* **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
* **Training Algorithm**: Group Relative Policy Optimization (GRPO)
* **Framework**: Google Tunix (JAX-native)
* **Compute**: Kaggle TPU (v5e-8)
* **Dataset**: GSM8K (grade-school math word problems)

GRPO improves reasoning by generating multiple candidate answers for each prompt and optimizing the model based on their **relative quality** using custom reward functions.

---

## 🎯 Training Objective

The model is trained to:

1. Produce reasoning inside `<reasoning>...</reasoning>`
2. Produce a final numeric answer inside `<answer>...</answer>`
3. Maximize correctness while staying close to the base model (KL regularization)

---

## 🧪 Reward Functions

The training pipeline uses four reward components:

* **Exact format match** – rewards outputs that perfectly follow the required tag structure
* **Approximate format match** – rewards partial compliance
* **Answer correctness** – rewards exact or near-correct answers
* **Number extraction** – rewards valid numeric answers even when embedded in text

These are combined inside GRPO to guide learning.

---

## 📊 Results

After fine-tuning on GSM8K:

* **Format accuracy**: ~46%
* **Exact numerical accuracy**: ~14.5%
* **Partial numerical accuracy**: ~16%

The key outcome is a large improvement in **structured reasoning output**, which was the primary goal of the hackathon.

---

## 🧩 Example Output

```
<reasoning>
The car travels 120 km in 2 hours. Speed is distance divided by time.
120 ÷ 2 = 60.
</reasoning>
<answer>
60
</answer>
```

---

## 🛠 How to Run

This project is designed to run on **Kaggle with TPU enabled**.

Steps:

1. Open the notebook in Kaggle
2. Enable TPU (v5e-8)
3. Run all cells in order
4. Train the model using GRPO
5. Evaluate using the built-in evaluation code

Checkpoints are saved automatically and can be reloaded for evaluation.

---

## 📦 What’s in this Repo

* Kaggle notebook with:

  * Dataset loading
  * Prompt formatting
  * Reward functions
  * GRPO training loop
  * Evaluation pipeline
* Configuration for:

  * LoRA
  * GRPO
  * Checkpointing
  * Inference

---

## 🔬 Why This Matters

This project demonstrates how **reinforcement learning–based post-training** can be used to make open-weight language models more **transparent, trustworthy, and interpretable**. It shows that even small models trained with limited compute can learn to reason in structured, human-readable ways.

---

## 🏆 Hackathon

This project was built as a submission for the **Google Tunix Hackathon: Train a Model to Show Its Work**.

---


