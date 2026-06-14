# DermArbiter: Multi-Agent Collaborative Scaffolding for Dermatological Decision Support
**Target Journal:** Nature Medicine (Preprint / Article)  
**Authors:** Furkan Ahi, Mahmut Emre Karaman, Dr. Abdurrahim Yılmaz  

---

## Abstract
* **Background:** Clinician shortages in primary care lead to delayed skin cancer diagnoses. AI classifiers show high SOTA performance but fail to emulate the multi-dimensional, collaborative reasoning of a dermatologist panel, frequently neglecting patient history, failing standard coding (ICD/SNOMED), or exhibiting calibration and demographic bias.
* **Methods:** We introduce **DermArbiter**, an agentic framework structured via LangGraph. It coordinates a panel of four specialized LLM agents (Specialist, Generalist, Skeptic, Moderator) collaborating through a shared blackboard state machine. The panel proposes diagnostic tools (classifiers, RAG, clinical ontologies), critiques discrepancies, debates diagnostic uncertainty, and synthesizes a structured differential diagnosis mapped to ICD-10 and SNOMED-CT codes. We evaluate DermArbiter against standard baseline classifiers and frontier LLMs across three benchmark layers: (1) SOTA replication (DermAgent), (2) a novel multi-dimensional clinical benchmark (DermAbench), and (3) ablation analysis of agentic deltas.
* **Findings:** [Placeholder for evaluation stats: e.g., DermArbiter achieves superior Top-3 diagnostic accuracy compared to single LLMs and base classifiers. Crucially, the multi-agent consensus debate significantly improves Expected Calibration Error (ECE) and reduces accuracy gaps across Fitzpatrick skin type subgroups (I–VI) by X%, while providing 100% compliant ontology mappings.]
* **Interpretation:** Cooperative agentic architectures can bridge the gap between classification-only AI and holistically grounded clinical decision support.

---

## 1. Introduction
* **Clinical Gap:** The diagnostic accuracy of primary care providers for skin cancer is significantly lower than board-certified specialists. Although deep learning classifiers achieve high performance on curated datasets, they function as black boxes, cannot process complex, unstructured patient histories, and lack the ability to explain or contextualize their findings within medical guidelines.
* **Limitations of Single Frontier Models:** Large Language Models (e.g., GPT-4o) possess extensive clinical knowledge but fail in reasoning calibration, suffer from hallucinations in RAG retrieval, and cannot conform strictly to standard medical ontologies (ICD-10/SNOMED-CT) without structural constraints.
* **The DermArbiter Solution:** We present DermArbiter, a collaborative multi-agent architecture. By structuring debate, critiquing reasoning flaws (Skeptic agent), and integrating specialized tools, we emulate the workflow of a dermatological tumor board.

---

## 2. Results

### 2.1 Layer 1: DermAgent Benchmark Reproduction (Credibility Anchor)
* **Goal:** Prove DermArbiter matches or exceeds SOTA agents on traditional 7-class classification tasks (HAM10000, Derm7pt).
* **Key Figures:** 
  - **Table 1:** Comparative accuracy, sensitivity, and specificity of DermArbiter vs. DermAgent vs. single base models.

### 2.2 Layer 2: DermAbench Multi-Dimensional Clinical Evaluation
* **Goal:** Demonstrate agentic superiority on the 8 clinical dimensions defined in `DERMABENCH_PROTOCOL.md`.
* **Key Figures:**
  - **Figure 1:** Radial chart comparing DermArbiter vs. GPT-4o-single across Visual, Narrative, Coding, DDx Quality, Calibration, Fairness, Safety, and Grounding.
  - **Table 2:** ICD-10 and SNOMED-CT exact mapping accuracy, demonstrating the impact of `OntologyGraph` integration.

### 2.3 Layer 3: Ablation & Agentic Delta Analysis
* **Goal:** Prove that the improvement is caused by the *agentic scaffolding* (debate, tools, multi-agent collaboration) rather than just the underlying base model.
* **Key Figures:**
  - **Figure 2:** Ablation deltas ($\Delta Acc$, $\Delta ECE$, $\Delta Fairness$) showing the contribution of each stage (Base Model -> +Tools -> +Agents -> +Debate).

---

## 3. Discussion
* **Clinical Implications:** How DermArbiter can act as a reliable triaging assistant in primary care settings, catching malignant lesions (via the Skeptic agent) that might otherwise be overlooked.
* **AI Safety & Alignment:** The role of the Moderator agent and early-exit gating in enforcing safety ceilings, reducing token budget waste, and preventing diagnostic drift.
* **Limitations:** Dependency on base model capabilities; potential latency constraints for real-time consultation; the necessity of local quantization support on low-resource environments.

---

## 4. Methods
* **Blackboard State & LangGraph Orchestrator:** Detailed diagram and explanation of the 5-phase debate protocol (Plan & Probe, Independent Reading, Reveal & Critique, Targeted Debate, Synthesis).
* **Model Router & Fallback Chain:** Explanation of the `ModelRouter` backend handling Google Gemini, Local HuggingFace, and OpenAI GPT-4o models with exponential retry logic.
* **Diagnostic Tools Integration:** Descriptions of PanDerm, MAKE, DermoGPT, Guideline RAG, Case RAG, and OntologyGraph.
* **Statistical Methods:** Bootstrap confidence intervals, Expected Calibration Error formulas, and demographic parity distance metrics.
