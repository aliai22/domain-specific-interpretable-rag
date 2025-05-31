# Domain-Adaptive LLM Architectures: Finetuning, RAG Pipelines & Interpretability

## 📌 Project Description

With the rapid advancements in Large Language Models (LLMs), the need for **domain-specific adaptation** has become more critical than ever. General-purpose LLMs often fall short in specialized tasks due to a lack of contextual and domain-relevant understanding. This project lays the groundwork for adapting LLMs to specific domains through a combination of **model finetuning**, **Retrieval-Augmented Generation (RAG) architectures**, and **keyword-based interpretability techniques**.

We conduct a series of ablation studies to assess how different model finetuning strategies and RAG configurations perform across various information retrieval and generation tasks. Our goal is to explore practical and effective techniques to enhance LLM performance in specialized domains.

---

## 🗂️ Project Structure

```
├── finetuning/
│   ├── base_model/              # Scripts and configs for finetuning base LLMs
│   └── embedding_model/         # Scripts for training or finetuning embedding models
│
├── rag_pipelines/
│   ├── base_llm_db1/            # RAG pipeline: base LLM with database 1
│   ├── base_llm_db2/            # RAG pipeline: base LLM with database 2
│   ├── finetuned_llm_db1/       # RAG pipeline: finetuned LLM with database 1
│   └── finetuned_llm_db2/       # RAG pipeline: finetuned LLM with database 2
│
├── services/                    # Modular service classes for preprocessing, evaluation, etc.
├── utils/                       # Utility functions
├── requirements.txt             # Python package requirements
├── main.py                      # Entry point for running experiments
└── README.md
```

---

## ⚙️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/domain-adaptive-llm.git
   cd domain-adaptive-llm
   ```

2. **Create and activate a virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Features

- 🔧 **Finetuning Pipelines**: For both base and embedding models using supervised QA datasets.
- 🧠 **RAG Architectures**: Four distinct pipelines combining various models and database configurations.
- 📊 **Evaluation Metrics**: BLEU, ROUGE, and embedding-based similarity for measuring generation quality.
- 🧩 **Modular Design**: Easy to plug and play components like dataset preprocessing, model loading, and training.
- 🕵️‍♂️ **Interpretability Techniques** (coming soon): Keyword extraction and attribution-based insights.

---

## 📌 TODO / Coming Soon

- [ ] Add pretrained checkpoints for quick-start
- [ ] Include example datasets and usage demos
- [ ] Integrate Streamlit or Gradio app for interactive demo
- [ ] Add keyword attribution & visualization techniques
- [ ] Benchmark results & ablation summary

---

## 👤 Author

Developed and maintained by **[Your Name]**

---

## 📄 License

*To be added – MIT, Apache 2.0, or any preferred open-source license.*

---

## 📬 Contact

For issues or feature requests, feel free to open an issue on this repo or contact the author.