# DermAbench: Multi-Dimensional Agentic Dermatology Benchmark Protocol
**Version:** 1.0 (Lite)  
**Authors:** Furkan Ahi, Mahmut Emre Karaman  
**Clinical Advisor:** Dr. Abdurrahim Yılmaz  
**Date:** June 2026  

---

## 1. Executive Summary
DermAbench is a holistic evaluation suite designed to assess the clinical value of multi-agent cooperative workflows in dermatology. While traditional benchmarks restrict evaluation to isolated 7-class image classification (e.g., HAM10000), DermAbench measures clinical decision quality across **8 distinct dimensions**. 

This protocol acts as the blueprint for **Phase B (DermAbench Design & Curation)**. It establishes clinical validation requirements, patient demographic representation, scoring criteria, and reference baselines to prevent confounding.

---

## 2. The 8 Evaluation Dimensions & Mapped Components

| Dimension | Target Evaluation Area | Active System Component | Baseline comparison (GPT-4o single) |
| :--- | :--- | :--- | :--- |
| **1. Görsel Tanı (Visual Diagnosis)** | Accuracy on challenging/atypical clinical images. | `PanDerm`, `MAKE`, `DermoGPT` | Moderate |
| **2. Klinik Öykü Anlama (Narrative)** | Feature extraction from raw clinical histories. | LLM Reasoning & Synthesis | Low (No grounding) |
| **3. Standart Kodlama (Coding)** | Mappings to ICD-10/11 and SNOMED-CT codes. | `OntologyGraph` + Agent Briefs | Low (Frequent errors) |
| **4. Diferansiyel Kalitesi (DDx Quality)** | Top-$k$ rank ordering and clinical reasoning. | Debate + Synthesis Phase | Moderate |
| **5. Kalibrasyon (Calibration)** | Expected Calibration Error vs. confidence. | `UncertaintyProbe` | Low (Overconfident) |
| **6. Fairness (Equity)** | Stability across Fitzpatrick skin types I–VI. | `FairnessProbe` | Unmeasured / Biased |
| **7. Güvenlik ve Triyaj (Safety)** | Identification of critical cases (e.g., melanoma). | `Skeptic` + `Uncertainty` | Variable |
| **8. Kanıta Dayalılık (Grounding)** | Grounding clinical claims in retrieved literature. | `CaseRAG` + `GuidelineRAG` | Low (Hallucinations) |

---

## 3. Case Selection & Curation Criteria
DermAbench v1-lite consists of **150–250 curated cases** compiled from diverse public clinical repositories:
* **SCIN (Google):** For rich patient metadata, early-stage presentation, and diverse Fitzpatrick types.
* **Derm1M:** For raw clinical images paired with unstructured clinical history texts.
* **DDI (Stanford):** For evaluating fairness across dark skin tones (Fitzpatrick V–VI).
* **PubMed Case Reports:** For complex, atypical presentation narratives with ground-truth pathology labels and associated ICD/SNOMED codes.

### Inclusion Criteria:
1. Cases must contain both a clinical/dermoscopic image and an unstructured clinical description/history.
2. Cases must have a biopsy/histopathologically confirmed diagnosis as ground truth.
3. Fitzpatrick skin type must be annotated or extractable via `FairnessProbe`.

---

## 4. Gold-Standard Annotation Workflow
To prevent self-serving evaluation bias, all cases undergo a strict pre-registration validation pipeline:
1. **Initial Extraction:** Automated extraction of clinical description, ground-truth label, and metadata.
2. **Clinical Mapping:** Standardizing ground-truth diagnoses to SNOMED-CT concepts and ICD-10 codes using clinical ontologies.
3. **Blind Clinical Review:** Dr. Abdurrahim Yılmaz performs a blinded review of the clinical descriptions and images to provide:
   - Ranked reference differentials.
   - Recommended management plans (biopsy vs. monitor vs. reassure).
4. **Consensus Freeze:** Once reviewed, case ground truths are locked and registered as the benchmark gold standard prior to running evaluations.

---

## 5. Scoring Formulas & Metrics Harness Design
The evaluation harness (`metrics.py` and `fairness_analyzer.py`) computes score aggregates per dimension:

### 1. Classification & Coding
* **Differential Top-3 Accuracy ($Acc@3$):** Proportion of cases where the biopsy-confirmed diagnosis is contained in the top-3 ranked differential list.
* **Coding Accuracy ($Acc_{code}$):** Exact matching of synthesized ICD-10 and SNOMED-CT codes to the gold-standard reference codes.
  $$Acc_{code} = \frac{\sum_{i=1}^{N} \mathbb{I}(code_{pred} == code_{true})}{N}$$

### 2. Calibration & Triyaj
* **Expected Calibration Error (ECE):**
  $$ECE = \sum_{m=1}^{M} \frac{|B_m|}{N} |acc(B_m) - conf(B_m)|$$
* **Melanoma Triyaj Sensitivity:** Sensitivity of triggering the "Skeptic urgent referral" flag for histopathologically malignant lesions.

### 3. Fairness
* **Maximum Group Accuracy Gap:**
  $$\Delta Acc = \max_{g \in G} Acc(g) - \min_{g \in G} Acc(g)$$
  Where $G$ represents Fitzpatrick subgroups (Light: I–III vs. Dark: IV–VI).

---

## 6. Confound Prevention & Baseline Strategy
To isolate the value of agentic scaffolding from base model capability, the evaluation uses a controlled comparative architecture:

1. **Base Model Baseline:** Running a single instance of the backend model (e.g., Gemini-2.5-flash) with zero tools and zero debate.
2. **External Frontier Reference:** A single instance of GPT-4o with zero tools.
3. **DermArbiter (Full):** The full 5-phase debate pipeline running on the same base models.

Ablation deltas ($\Delta Acc$, $\Delta ECE$, $\Delta Fairness$) will serve as primary figures in the paper, proving the value added by multi-agent collaboration and RAG grounding.
