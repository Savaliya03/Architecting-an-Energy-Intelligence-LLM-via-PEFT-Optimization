
# 📌 Project: Architecting an Energy Intelligence LLM via PEFT Optimization

An enterprise-grade implementation of a local, fine-tuned AI agent capable of translating natural language into highly accurate, domain-aware SQL queries. This project was developed to localize AI intelligence for industrial database management at Innovation Intellect.

> **📝 Note on Nomenclature & Weights:**
> This enterprise AI solution was developed by and belongs to **Innovation Intellect**.
> Please note that **InnovativeEngineers** is our official internal engineering namespace. You will strictly see `InnovativeEngineers` used in repository paths, API configurations, and our Hugging Face model registry. Both names represent the same corporate entity.
>
> This repository contains **ONLY THE CODE**. The merged model weights are hosted securely under our engineering namespace on Hugging Face: [InnovativeEngineers/Energy-Intelligence](https://huggingface.co/InnovativeEngineers/Energy-Intelligence)

---

## 🚀 1. Overview & Achievements

The core objective is to transform a standard Large Language Model into a specialized SQL agent. By utilizing a massive "Parent" model (DeepSeek V3.2 37B) to train a lightweight "Child" model (Qwen 2.5-7B-Instruct), we achieve high-end logic extraction without the associated hardware costs.

* **Logic Accuracy:** Increased from a base of 22% to 80% post-training.
* **Overall Improvement:** A 263% boost in reliable database reporting.
* **Schema Knowledge:** Evolved from hallucinating table names to using exact, site-specific tables.

---

## 💻 2. Installation & Quick Start

### Repository Structure
```text
.

├── .gitignore             # Security block for weights and private data
├── README.md              # This instruction file
├── requirements.txt       # Core ML dependencies
├── train.py               # The LoRA/PEFT fine-tuning script
├── inference.py           # The interactive CLI chat script
├── questions_and_answers.json  # Raw dataset / Q&A pairs
└── questions_and_queries.json  # Formatted Text-to-SQL training data

```

### Prerequisites & Setup

Ensure you have Python 3.8+ and Dual NVIDIA T4 GPUs (or equivalent VRAM).



#### 1. Clone the repository
```
git clone [https://github.com/YOUR_USERNAME/EnergyIntelligence-SQL-Agent.git](https://github.com/YOUR_USERNAME/EnergyIntelligence-SQL-Agent.git)
cd EnergyIntelligence-SQL-Agent
```

#### 2. Install required ML dependencies

```
pip install -r requirements.txt

```

* * * * *

💬 3. Execution Flow (Running the Agent)
----------------------------------------

To interact with the SQL Agent, run the inference script. This script connects directly to the merged model hosted on Hugging Face.




Bash

```
python inference.py

```

**Operational Steps:**

1.  **Input:** You will be prompted to enter a natural language question (e.g., *"Show me energy consumption for last week"*).

2.  **Processing:** The script enforces strict system guardrails and passes the prompt to the Qwen 2.5-7B logic layer.

3.  **Output:** The system generates a validated, read-only SQL `SELECT` query ready for database execution.

* * * * *

🧠 4. Model Training (Replication)
----------------------------------

To replicate the 4-Bit Quantization and LoRA fine-tuning process locally (requires the `qustions_and_quaries.json` dataset in the root directory):


Bash

```
python train.py

```

*Outputs and adapter weights will be saved in a local `./qwen-chat-finetune` directory. This training methodology forces the AI to learn underlying SQL logic rather than memorizing data, masking instruction loss during execution.*

* * * * *

🛡️ 5. Features & Security Guardrails
-------------------------------------

This project offers powerful data retrieval capabilities locked down by strict security measures:

-   **Strict Read-Only Execution:** The agent is programmatically restricted to data retrieval (`SELECT` statements only).

-   **Zero Manipulation Risk:** Instantly aborts and refuses any prompts attempting to execute `DELETE`, `UPDATE`, `INSERT`, or `DROP` commands.

-   **Time-Series & Statistical Analysis:** Capable of analyzing data trends over hours/days/months, comparing parameters (e.g., Phase currents), and identifying statistical deviations (e.g., standard deviations).

* * * * *

💼 6. Business Impact & Cost-Efficiency
---------------------------------------

-   **Reduced Overhead:** Eliminates the need for technical teams to manually draft routine database queries.

-   **Self-Service Analytics:** Enables non-technical staff to pull instant data using plain English questions.

-   **High ROI:** Captures ~80% of the reasoning capabilities of a massive 37B model but runs on significantly cheaper hardware.

* * * * *

🗺️ 7. Future Roadmap
---------------------

This architecture acts as a scalable foundation for future enhancements:

-   **Version 2.0 (Data Visualization):** Integration of Natural Language-to-Chart capabilities for direct visual insights.

-   **Enterprise Integration:** Seamless connectivity with Business Intelligence tools such as Power-BI, Tableau, and Microsoft Excel.

-   **Zero-Shot Expansion:** Enhancing the model's reasoning to handle diverse and unseen database schemas without intensive re-training.

* * * * *

🙌 Acknowledgements
-------------------

Special thanks to the **Qwen Team** (Alibaba Cloud) for releasing the Qwen 2.5-7B-Instruct model, and the **DeepSeek Team** for the DeepSeek V3.2 model. Their commitment to open-source AI provided the foundational intelligence that made this Text-to-SQL fine-tuning and optimization project possible.

* * * * *

📜 Confidentiality & Copyright
------------------------------

© 2026. All rights reserved. This framework and documentation are proprietary to **Innovation Intellect Pvt. Ltd.** No part of this repository may be reproduced, distributed, or transmitted in any form without prior written permission. Unauthorized reproduction or distribution is strictly prohibited.
