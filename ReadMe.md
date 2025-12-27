## Repository Structure

### `BpmnAnalyser/`

Java-based tool used to compute **Output Metrics (OM)** by comparing generated BPMN models against expert reference models.

**Functionality:**
- Parses BPMN XML files  
- Computes structural and quality metrics (e.g. precision, recall, element counts, error measures)  
- Exports computed metrics as CSV files  

**⚠️ Limitations**

This project cannot be fully re-run, as expert reference BPMN models are not publicly available.  
However, all generated CSV outputs used in the thesis are included.

---

### `BPMNPredictor/`

Python-based implementation of the prediction experiments described in the thesis.

**Functionality:**
- Loads PDM, OM, and EM data  
- Trains multi-output regression models  
- Produces prediction results for research questions (**RQ1, RQ2, RQ4, RQ5**)  

The entry point is `main.py`.  
All final results are written to the `results/` folder inside this project.

---

### `BPMNPredictor_Inputs_PDM_OM_EM/`

Contains the input datasets used by the prediction models:

- **Process Description Metrics (PDM)**  
- **Output Metrics (OM)**  
- **Expert Metrics (EM)**  

All data is anonymised and provided as CSV files.  
These files represent the **exact inputs used for the experiments reported in the thesis**.

---

### `GeneratedModels_A2/`

Contains the generated BPMN models for anonymisation level **A2**.

- Models are provided in BPMN XML format  
- Used for qualitative inspection and transparency  
- Corresponds to the strongest anonymisation setting discussed in the thesis  

Only A2 models are included, as lower anonymisation levels contain sensitive information that cannot be made public.

---

### `Predictor_Outputs/`

Contains the final outputs of the prediction experiments, including:

- R² and RMSE values  
- Per-target prediction results  
- Feature selection outcomes  

These files correspond directly to the results tables and figures reported in the thesis.  
No sensitive or confidential information is contained in this folder.

---

### `Process_Descriptions_A2/`

Contains the **textual process descriptions** used as input for BPMN generation at anonymisation level **A2**.

- Descriptions are fully anonymised  
- Provided in plain text format  
- Correspond to the BPMN models in `GeneratedModels_A2/`  

These descriptions represent the **input side of the modelling pipeline** under the strongest anonymisation setting.  
Lower anonymisation levels (A0, A1) are not included due to confidentiality constraints.
