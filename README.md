<h1 align="center">Tech Interview Cheatsheets: The "Gold Edition" 🚀</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Status-Active-success.svg" alt="Status">
  <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License">
  <img src="https://img.shields.io/badge/PRs-Welcome-brightgreen.svg" alt="PRs">
  <img src="https://img.shields.io/badge/Format-Interactive_Markdown-orange.svg" alt="Format">
</p>

<p align="center">
  <strong>The Ultimate Interactive Collection of Deep-Dive, Edge-Case Interview Questions for Senior AI & Data Roles</strong>
</p>

---

## 📖 About This Repository

Standard interview prep guides are filled with cliches: *"What is a CNN?"*, *"How does Backprop work?"*, *"What is a Left Join?"*. **This repository is different.**

This is a highly curated collection of **Edge-Case, Pathology, and Advanced Systems questions** asked in senior-level interviews at top-tier tech companies. It is designed to test the absolute boundaries of your understanding of modern architectures, distributed training dynamics, generative models, and statistical paradoxes.

### ✨ Now Fully Interactive!
This repository acts as an **Interactive Flashcard System**. All questions are hidden inside collapsible Markdown sections. 
1. Read the question.
2. Try to answer it in your head.
3. **Click on the question** to reveal the deep-dive technical answer!

---

## 🧠 The Holy Trinity of AI Roles

### 1. [Deep Learning: The "Edge-Case" Guide](./Deep_Learning_Interview_Notes.md)
*Focuses on modern LLMs, Generative AI, and distributed training mechanics.*
- **Training Pathology:** Double Descent, Grokking, Lottery Ticket Hypothesis, Dying ReLU.
- **LLM Mechanics:** Speculative Decoding, KV Caching, FlashAttention (Tiling), RoPE vs Absolute embeddings, SwiGLU.
- **Optimization:** Why `bfloat16` dominates `float16`, AdamW Weight Decay vs Adam, AMSGrad.
- **Generative AI:** Diffusion noise prediction vs image prediction, Classifier-Free Guidance (CFG), DPO vs PPO for RLHF.
- **Distributed Training:** ZeRO-3 / FSDP vs DDP, Gradient Checkpointing.

### 2. [Data Analytics & Data Science: The "Edge-Case" Guide](./Data_Analytics_Interview_Notes.md)
*Focuses on statistical paradoxes, causal inference, and silent SQL/Pandas killers.*
- **Statistical Traps:** Simpson's Paradox, The Peeking Problem in A/B tests, Multiple Comparisons, Will Rogers Phenomenon, Goodhart's Law.
- **Advanced SQL:** Recursive CTEs, Window Functions without `ORDER BY`, The Cartesian Fan-Out explosion.
- **Pandas Mechanics:** The `SettingWithCopyWarning` memory trap, Categorical memory optimization, the Python GIL.
- **Causal Inference & ML:** Quasi-experiments (RDD, Propensity Matching), SMOTE Data Leakage, Curse of Dimensionality.
- **Bonus:** How to pass Automated AI Video Interviews (HireVue).

### 3. [MLOps & Data Engineering: The "Edge-Case" Guide](./MLOps_Data_Engineering_Interview_Notes.md)
*Focuses on distributed systems, streaming architectures, and serving models in production.*
- **Apache Kafka & Spark:** Partition overload, ZooKeeper bottlenecks, Shuffle OOM errors, Data Skew (Salting).
- **Kubernetes & Serving:** Pipeline Parallelism vs Data Parallelism, HPA Cold Starts with KEDA, Thundering Herd caching problems.
- **Orchestration:** Airflow global scope parsing errors, Idempotency in ETL.
- **Production ML:** Concept Drift vs Data Drift, Proxy Metrics for Delayed Ground Truth.

---

## 🤝 Contributing
Found a crazy edge-case question in a FAANG interview? We'd love to add it! Please read the [CONTRIBUTING.md](./CONTRIBUTING.md) guide for details on our code of conduct, and the process for submitting pull requests.

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.
