import pandas as pd
from src.data_loader import load_pdm, load_om, \
    load_em, combine_experts
from src.evaluate_anonLevels import \
    print_anonymization_effects
from src.evaluate_output import \
    generate_and_save_report
from src.evaluate_system import \
    print_system_evaluation
from src.om_a0_pdm_predict_em_a1_a2 import \
    print_om_a0_pdm_predict_em_a1_a2
from src.om_predict_em import print_om_predict_em
from src.pdm_om_predict_em import \
    print_pdm_om_predict_em

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)




# Load all data
pdm_path = "data/Data_ProcessDescriptions/Process_Descriptions.csv"
om_root = "data/Data_Output"
em_root = "data/Data_Experts"
base_dir = "C:/Dev/BPMNPredictor/results"

pdm = load_pdm(pdm_path)
om = load_om(om_root)
em = load_em(em_root)

em_combined = combine_experts(em)

#RQ1
print_om_predict_em(om, em_combined)

#RQ2
print_pdm_om_predict_em(om, em_combined, pdm)

#RQ4
print_anonymization_effects(om, em_combined)

#RQ5
print_om_a0_pdm_predict_em_a1_a2(om, em_combined, pdm)


#Personal
#print_system_evaluation(om, em_combined)

