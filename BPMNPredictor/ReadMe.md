# BPMNPredictor

This project contains the prediction models used in the thesis to estimate **expert ratings of LLM-generated BPMN models** based on **Process Description Metrics (PDM)** and **Output Metrics (OM)**.

---

## Purpose

The **BPMNPredictor** implements all regression experiments described in the thesis.  
It loads precomputed metrics, trains predictive models, and generates evaluation results for the individual research questions (**RQ1–RQ5**).

The implementation follows **exactly** the experimental setup reported in the thesis.

---

## Repository Structure (Relevant Parts)



### `src/`
Contains all source code for:

- Data loading
- Feature selection
- Model training
- Model evaluation

### `data/`
Contains the CSV input data used for prediction:

- **Process Description Metrics (PDM)**
- **Output Metrics (OM)**
- **Expert Ratings (EM)**

> **Note:** Only anonymised data is included.  
> Sensitive expert reference data and non-anonymised inputs cannot be made public.

### `results/`
Contains all generated outputs (CSV files and reports).  
These files represent the **complete prediction results used in the thesis** and do **not contain sensitive information**.

### `main.py`
Entry point for running all prediction experiments.

---

## Running the Experiments

All experiments are executed by running:

python `main.py`


The `main.py` script performs the following steps:

1. Loads PDM, OM, and expert ratings (EM) from the `data/` directory  
2. Combines expert ratings where applicable  
3. Executes prediction tasks corresponding to the research questions  
4. Writes all results to the `results/` directory  

The research questions are explicitly marked in the code:

- **RQ1** – OM → EM prediction  
- **RQ2** – PDM + OM → EM prediction  
- **RQ4** – Effect of anonymisation on prediction  
- **RQ5** – Cross-anonymisation prediction  

---


## Notes on Reproducibility

The project cannot be fully re-run as originally executed, because:

- Only anonymised input data is included  
- Expert reference BPMN models are not publicly available  

Nevertheless, all final prediction outputs used in the thesis are included in `results/`, allowing full inspection and verification of the reported findings.

---

## Folder Naming in Results

The result folders correspond to research questions but may not be sequentially numbered due to iterative development:

- **RQ1** → RQ2  
- **RQ2** → RQ5  
- **RQ4** → RQ3  
- **RQ5** → RQ4  

These mappings are documented here for clarity and do not affect the correctness of the reported results.





