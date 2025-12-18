
import os
from typing import List, Dict
import pandas as pd

from src.Schreiben import \
    evaluate_single_value_baselines
from src.Thesis_Visuals import plot_em_boxplots, \
    plot_om_boxplots, summarize_om_metrics, \
    plot_pdm_boxplots, summarize_pdm_metrics, \
    summarize_em_metrics
from src.data_loader import load_em, load_om, \
    load_pdm, combine_experts, \
    create_combined_key_om, \
    create_combined_key_em_combined
from src.evaluate_anonLevels import \
    print_anonymization_effects
from src.evaluate_output import \
    parse_model_outputs, \
    parse_multiple_rq_outputs, \
    generate_and_save_report
from src.evaluate_system import \
    print_system_evaluation
from src.om_a0_pdm_predict_em_a1_a2 import \
    combine_id_om_without_anon, \
    combine_id_em_without_anon, \
    print_om_a0_pdm_predict_em_a1_a2, \
    combine_pdm_om_features
from src.om_predict_em import print_om_predict_em
from src.pdm_om_predict_em import \
    print_pdm_om_predict_em, \
    combine_pdm_om_with_anon
from src.pdm_predict_om_em import \
    print_pdm_predict_em, \
    create_averaged_em_combined

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

em_root = "C:/Dev/BPMNPredictor/data/Data_Experts"
em = load_em(em_root)

"""
# Inspect structure and values
for expert, anon_dict in em.items():
    print(f"\n🔬 Expert: {expert}")
    for level, df in anon_dict.items():
        print(f"  📦 AnonLevel: {level}")

        # Get one-hot encoded columns
        group_cols = [col for col in df.columns if col.startswith("Group_")]
        anon_cols = [col for col in df.columns if col.startswith("AnonLevel_")]

        for group_col in group_cols:
            llm_name = group_col.replace("Group_", "")
            llm_df = df[df[group_col] == 1]
            print(f"    🧠 LLM: {llm_name}")
            print(llm_df.head(50))
            print(f"    ➖ Columns: {list(llm_df.columns)}")
            print(f"    🔢 Group one-hot: \n{llm_df[group_cols].head(3)}")
            print(f"    🔢 AnonLevel one-hot: \n{llm_df[anon_cols].head(3)}\n")

"""
em_combined = combine_experts(em)
"""
for anon_level, df in em_combined.items():
    print(f"\n📦 Combined AnonLevel: {anon_level}")

    group_cols = [col for col in df.columns if col.startswith("Group_")]
    anon_cols = [col for col in df.columns if col.startswith("AnonLevel_")]

    for group_col in group_cols:
        llm_name = group_col.replace("Group_", "")
        llm_df = df[df[group_col] == 1]
        print(f"  🧠 LLM: {llm_name}")
        print(llm_df.head(3))  # show sample rows
        print(f"  ➖ Columns: {list(llm_df.columns)}")
        print(f"  🔢 Group one-hot:\n{llm_df[group_cols].head(3)}")
        print(f"  🔢 AnonLevel one-hot:\n{llm_df[anon_cols].head(3)}\n")


for llm, subgroups in om.items():
    print(f"\n🧠 LLM: {llm}")
    for subgroup, df in subgroups.items():
        print(f"  📦 AnonLevel (raw): {subgroup}")

        group_cols = [col for col in df.columns if col.startswith("Group_")]
        anon_cols = [col for col in df.columns if col.startswith("AnonLevel_")]

        print(df.head(3))
        print(f"  ➖ Columns: {list(df.columns)}")
        print(f"  🔢 Group one-hot:\n{df[group_cols].head(3)}")
        print(f"  🔢 AnonLevel one-hot:\n{df[anon_cols].head(3)}\n")


pdm_path = "C:/Dev/BPMNPredictor/data/Data_ProcessDescriptions/Process_Descriptions.csv"
pdm_df = load_pdm(pdm_path)
print("📄 PDM Data Loaded:")
print(pdm_df.head(48))
print("🧾 Columns:", list(pdm_df.columns))

averaged = create_averaged_em_combined(em_combined)
for level, df in averaged.items():
    print(f"\n🧪 Averaged EM Data for AnonLevel: {level}")
    print(df.head())
"""

pdm_path = "C:/Dev/BPMNPredictor/data/Data_ProcessDescriptions/Process_Descriptions.csv"
pdm = load_pdm(pdm_path)
om_root = "C:/Dev/BPMNPredictor/data/Data_Output"  # or absolute path if needed
om = load_om(om_root)



'''
for llm, subgroups in om.items():
    print(f"\n🧠 LLM: {llm}")
    for subgroup, df in subgroups.items():
        print(f"  📦 AnonLevel (raw): {subgroup}")

        group_cols = [col for col in df.columns if col.startswith("Group_")]
        anon_cols = [col for col in df.columns if col.startswith("AnonLevel_")]

        print(df.head(20))
        print(f"  ➖ Columns: {list(df.columns)}")
        print(f"  🔢 Group one-hot:\n{df[group_cols].head(3)}")
        print(f"  🔢 AnonLevel one-hot:\n{df[anon_cols].head(3)}\n")


from data_loader import create_combined_key_em_combined

# Assuming em_combined is already loaded via combine_experts
em_combined_with_key = create_combined_key_em_combined(em_combined)

print("✅ EM Combined Keys Generated\n")

for level, df in em_combined_with_key.items():
    print(f"🔍 AnonLevel: {level}")
    print(f"🔑 Sample combined_id index:\n{df.index[:5]}")
    print(f"🧠 Sample rows:\n{df.head(100)}\n")

om_combined = create_combined_key_om(om)

for anon_level, df in om_combined.items():
    print(f"\n📦 AnonLevel: {anon_level}")
    print("🔑 Index preview:", df.index[:5].tolist())
    print("📄 Columns:", df.columns.tolist())
    print(f"🧠 Sample rows:\n{df.head(100)}\n")


# Convert to keyed format
om_keyed = create_combined_key_om(om)

# Combine
dm_df_indexed = pdm.set_index("process_id")
combined_df = combine_pdm_om_with_anon(dm_df_indexed, om_keyed)

print("✅ Combined shape:", combined_df.shape)
print("🧾 Sample rows:")
print(combined_df.head())
'''

#print_om_predict_em(om, em_combined)
#print_pdm_predict_em(em_combined, pdm)
#print_om_predict_em(em_combined, om)

'''
# Test AnonLevel None for both
anon_level = "None"
om_df = combine_id_om_without_anon(om, anon_level)
em_df = combine_id_em_without_anon(em_combined, anon_level)

print("\n✅ OM Combined (AnonLevel=None):")
print(om_df.head(80))
print(f"OM Index Sample: {om_df.index[:3].tolist()}")

print("\n✅ EM Combined (AnonLevel=None):")
print(em_df.head(80))
print(f"EM Index Sample: {em_df.index[:3].tolist()}")

print("\n🔍 Common Keys Between OM and EM:")
print(om_df.index.intersection(em_df.index).tolist())



om_df = combine_id_om_without_anon(om, anon_level="None")
pdm_df = pdm.set_index("process_id")

combined_df = combine_pdm_om_features(om_df, pdm_df)
print("✅ Combined OM + PDM features shape:", combined_df.shape)
print(combined_df.head(100))


#print_om_a0_pdm_predict_em_a1_a2(om, em_combined, pdm)

#print_pdm_om_predict_em(om, em_combined, pdm)
#print("🔍 PDM index after normalization:", pdm.index.tolist())

om_keyed = create_combined_key_om(om)

# Align and merge PDM into keyed OM
combined = combine_pdm_om_with_anon(pdm, om_keyed)

print("✅ Combined shape:", combined.shape)
print("🔍 Combined sample:")
print(combined.head(100))

print_pdm_om_predict_em(om, em_combined, pdm)
print_anonymization_effects(om, em_combined)


output_dir = "C:/Dev/BPMNPredictor/results/RQ1"

parsed = parse_model_outputs(output_dir)

print(f"✅ Parsed {len(parsed)} output files.\n")

for model in parsed:
    print(f"📄 File: {model.get('file')}")
    print(f"  AnonLevel: {model.get('AnonLevel', 'N/A')}")
    print(f"  Final R²: {model.get('Final_R2', 'N/A')}")
    print(f"  Final RMSE: {model.get('Final_RMSE', 'N/A')}\n")

    if model.get("Best_Model_Features"):
        print("  🌟 Best Model Features:")
        for feat in model["Best_Model_Features"]:
            print(f"    - {feat}")
        print()

    for section in [
        "Removed_Features_History",
        "Final_Feature_Weights",
        "R2_per_Target",
        "RMSE_per_Target"
    ]:
        if model.get(section):
            print(f"  📊 {section.replace('_', ' ')}:")
            for key, val in model[section].items():
                print(f"    - {key}: {val}")
            print()

    print("-" * 60)


base_dir = "C:/Dev/BPMNPredictor/results"  # adjust as needed
parsed_outputs = parse_multiple_rq_outputs(base_dir)

print(f"\n✅ Parsed {len(parsed_outputs)} model output files from RQ1, RQ2, RQ4, RQ5\n")

for model in parsed_outputs:
    print(f"📄 File: {model.get('file')}")
    print(f"  AnonLevel: {model.get('AnonLevel', 'N/A')}")
    print(f"  Final R²: {model.get('Final_R2', 'N/A')}")
    print(f"  Final RMSE: {model.get('Final_RMSE', 'N/A')}")

    if "Best_Model_Features" in model:
        print(f"  🌟 Best Model Features ({len(model['Best_Model_Features'])}):")
        for feat in model["Best_Model_Features"]:
            print(f"    - {feat}")

    # Print a preview of each metric dictionary
    for section in [
        "Removed_Features_History",
        "Final_Feature_Weights",
        "R2_per_Target",
        "RMSE_per_Target"
    ]:
        if section in model:
            print(f"  📊 {section}: {len(model[section])} entries")

    print("-" * 60)
'''
#print_pdm_om_predict_em(om, em_combined, pdm)
#base_dir = "C:/Dev/BPMNPredictor/results"  # adjust to your actual path
#generate_and_save_report(base_dir)

#print_system_evaluation(om, em_combined)

#print_averages() (normalization part has to be commented out)

#print_system_evaluation(om, em_combined)

#plot_em_boxplots(em_combined)
#plot_om_boxplots(om)

#summarize_om_metrics(om_root)

#plot_pdm_boxplots(pdm)

#summarize_pdm_metrics(pdm)

#summarize_em_metrics(em)


#RQ1
#print_pdm_predict_em(em_combined, pdm)

#RQ2old, RQ1New
#print_om_predict_em(om, em_combined)

#RQ3old, RQ4New
#print_anonymization_effects(om, em_combined)

#RQ4old, RQ5New
print_om_a0_pdm_predict_em_a1_a2(om, em_combined, pdm)

#RQ5old, RQ2New
#print_pdm_om_predict_em(om, em_combined, pdm)

#RQ6Old, RQ3New
#base_dir = "C:/Dev/BPMNPredictor/results"
#generate_and_save_report(base_dir)

#Personal
#print_system_evaluation(om, em_combined)

#schreiben
#evaluate_single_value_baselines(em_combined)
