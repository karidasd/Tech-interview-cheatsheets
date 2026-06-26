<h1 align="center">Tech Interview Cheatsheets: The "Gold Edition" 🚀</h1>

<p align="center">
  <strong>The Ultimate Collection of Deep-Dive, Edge-Case Interview Questions for Senior AI & Data Roles</strong>
</p>

---

## 📖 About This Repository

Standard interview prep guides are filled with cliches: *"What is a CNN?"*, *"How does Backprop work?"*, *"What is a Left Join?"*. **This repository is different.**

This is a highly curated collection of **Edge-Case, Pathology, and Advanced Systems questions** asked in senior-level interviews at top-tier tech companies. It is designed to test the absolute boundaries of your understanding of modern architectures, distributed training dynamics, generative models, and statistical paradoxes.

---

## 🧠 What's Inside?

### 1. [Deep Learning: The "Edge-Case" Guide](./Deep_Learning_Interview_Notes.md)
*Focuses on modern LLMs, Generative AI, and distributed training mechanics.*
- **Training Pathology:** Double Descent, Grokking, Lottery Ticket Hypothesis.
- **LLM Mechanics:** Why RMSNorm over LayerNorm? KV Caching, FlashAttention (Tiling), RoPE vs Absolute embeddings, SwiGLU.
- **Optimization:** Why `bfloat16` dominates `float16`, AdamW Weight Decay decoupling, Warmup schedules.
- **Generative AI:** Diffusion noise prediction vs image prediction, Classifier-Free Guidance (CFG), DPO vs PPO for RLHF.
- **Distributed Training:** ZeRO-3 / FSDP vs DDP, Gradient Checkpointing.

### 2. [Data Analytics & Data Science: The "Edge-Case" Guide](./Data_Analytics_Interview_Notes.md)
*Focuses on statistical paradoxes, causal inference, and silent SQL/Pandas killers.*
- **Statistical Traps:** Simpson's Paradox, The Peeking Problem in A/B tests, Multiple Comparisons (Bonferroni), Survivorship Bias.
- **Advanced SQL:** Recursive CTEs, Window Functions (RANK vs DENSE_RANK), The Cartesian Fan-Out explosion, NULL comparison behaviors.
- **Pandas Mechanics:** The `SettingWithCopyWarning` memory trap, Categorical memory optimization.
- **Causal Inference & ML:** Quasi-experiments (RDD, Propensity Matching), SMOTE Data Leakage in Cross-Validation, Curse of Dimensionality in Clustering.

---

## 🎯 How to Use

These files are written in **Markdown**, which means they render beautifully right here on GitHub. You do not need to download a PDF reader. 

Click on any of the guides above to read them directly, or clone the repository to study locally:

```bash
git clone https://github.com/karidasd/Tech-interview-cheatsheets.git
```

---
*Disclaimer: For educational purposes. Master these concepts not just to pass interviews, but to become a better, more dangerous engineer.*
