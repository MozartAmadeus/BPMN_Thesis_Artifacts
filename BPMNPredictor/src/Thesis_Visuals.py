import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("TkAgg")  # or "Agg" if you're running headless

import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import combine_experts, load_om


def summarize_om_metrics(root_dir: str):
    om_data = load_om(root_dir)
    all_frames = []

    for llm_dict in om_data.values():
        for df in llm_dict.values():
            all_frames.append(df)

    all_om = pd.concat(all_frames, ignore_index=True)

    exclude_cols = [
        'process_id', 'HasErrors', 'Group_Prototype', 'AnonLevel_None',
        'Group_Aalst', 'Group_Aau', 'AnonLevel_Anon1', 'AnonLevel_Anon2'
    ]
    #and not col.endswith('_fitness') and not col.endswith('_precision')
    # and (col.endswith('_fitness') or col.endswith('_precision'))
    om_numeric_cols = [col for col in all_om.select_dtypes(include='number').columns if col not in exclude_cols and not col.endswith('_fitness') and not col.endswith('_precision') and not col.startswith('HasErrors')]

    summary = all_om[om_numeric_cols].agg(['mean', 'median', 'std', 'min', 'max'], skipna=True).T
    summary.round(3)
    print(summary.to_string())



def plot_em_boxplots(em_combined: dict):
    target_columns = ['Grade', 'QU1', 'QU2', 'QU3']
    anon_levels = ['None', 'Anon1', 'Anon2']

    all_frames = []
    for level in anon_levels:
        df = em_combined.get(level)
        if df is not None:
            df = df.copy()
            df['Subgroup'] = f"A{anon_levels.index(level)}"
            all_frames.append(df)

    if not all_frames:
        print("No EM data available to plot.")
        return

    full_df = pd.concat(all_frames, ignore_index=True)

    melted = full_df.melt(id_vars=['Subgroup'], value_vars=target_columns, var_name='Metric', value_name='Score')

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=melted, x='Subgroup', y='Score', hue='Metric')
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
    plt.xlabel('Anonymization Level')
    plt.ylabel('Score')
    plt.title('EM Metric Distributions by Anonymization Level')
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_om_boxplots(om_data):
    # Flatten all dataframes by AnonLevel
    all_frames = []
    for llm, llm_dict in om_data.items():
        for subgroup, df in llm_dict.items():
            df = df.copy()
            df['Subgroup'] = subgroup
            df['LLM'] = llm
            all_frames.append(df)
            all_frames.append(df)



        full_df = pd.concat(all_frames, ignore_index=True)

    # Log any values > 1.0 in fitness/precision columns
    for col in full_df.columns:
        if col.endswith('_fitness') or col.endswith('_precision'):
            mask = full_df[col] > 1
            if mask.any():
                offenders = full_df[mask]
                for _, row in offenders.iterrows():
                    print(f"⚠️ {col} > 1.0 → value={row[col]:.3f}, Subgroup={row.get('Subgroup')}, LLM={row.get('LLM')}")

    # Filter fitness and precision columns
    # Filter fitness and precision columns
    metric_cols = [
        col for col in full_df.columns
        if ( col.endswith('_fitness') or  col.endswith('_precision') and not col.endswith("Errors") and not col.startswith("AnonLevel_")and not col.startswith("Group_")) and pd.api.types.is_numeric_dtype(full_df[col])
    ]
    target_metrics = sorted(metric_cols)
    anon_levels = ['A0', 'A1', 'A2']

    melted = full_df.melt(id_vars=['Subgroup'], value_vars=target_metrics, var_name='Metric', value_name='Score')
    melted['Metric'] = melted['Metric'].str.replace('_fitness', '_recall')
    plt.figure(figsize=(12, max(6, len(target_metrics) * 0.5)))
    sns.boxplot(data=melted, x='Subgroup', y='Score', hue='Metric')
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
    plt.xlabel('Anonymization Level')
    plt.ylabel('Score')
    plt.title('OM Structural Metric Distributions by Anonymization Level')
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_pdm_boxplots(pdm_df: pd.DataFrame):
    excluded = {'process_id'}
    bool_cols = pdm_df.select_dtypes(include='bool').columns.tolist()
    numeric_cols = [col for col in pdm_df.select_dtypes(include='number').columns if col not in excluded and col not in bool_cols]

    if not numeric_cols:
        print("No numeric columns available in PDM data.")
        return

    melted = pdm_df.melt(id_vars=[], value_vars=numeric_cols, var_name='Metric', value_name='Score')
    melted['Metric'] = melted['Metric'].str.replace('PDM', '')
    plt.figure(figsize=(12, max(6, len(numeric_cols) * 0.5)))
    sns.boxplot(data=melted, x='Metric', y='Score')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Metric')
    plt.ylabel('Score')
    plt.title('PDM Metric Distributions')
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def summarize_pdm_metrics(pdm_df: pd.DataFrame):
    excluded = {'process_id'}
    bool_cols = pdm_df.select_dtypes(include='bool').columns.tolist()
    numeric_cols = [
        col for col in pdm_df.columns
        if pd.api.types.is_numeric_dtype(pdm_df[col]) and col not in excluded and col not in bool_cols
    ]

    # Print process IDs
    print("🔍 All process_ids in PDM:")
    print(pdm_df['process_id'].unique())

    # Compute and print summary
    summary = pdm_df[numeric_cols].agg(['mean', 'median', 'std', 'min', 'max'], skipna=True).T
    summary = summary.round(3)
    print("📊 PDM Metric Summary:")
    print(summary.to_string())


def summarize_em_metrics(em_data: dict):
    target_columns = ['Grade', 'QU1', 'QU2', 'QU3']
    all_boxplot_frames = []
    print("📊 EM Metric Summary by Expert and AnonLevel:")
    for expert_id, anon_dict in em_data.items():
        print(f"Expert: {expert_id}")
        for anon_level in ['None', 'Anon1', 'Anon2']:
            df = anon_dict.get(anon_level)
            if df is None or not isinstance(df, pd.DataFrame):
                print(f"  AnonLevel {anon_level}: No data")
                continue

            missing_cols = [col for col in target_columns if col not in df.columns]
            if missing_cols:
                print(f"  AnonLevel {anon_level}: Missing columns {missing_cols}")
                continue

            values = df[target_columns].values.flatten()
            mean_val = round(values.mean(), 3)
            median_val = round(np.median(values), 3)
            std_val = round(values.std(), 3)
            min_val = round(values.min(), 3)
            max_val = round(values.max(), 3)
            spread = round(df[target_columns].mean().std(), 3)

            print(f"AnonLevel: {anon_level}")
            print(f"    Mean:   {mean_val}")
            print(f"    Median: {median_val}")
            print(f"    Std:    {std_val}")
            print(f"    Min:    {min_val}")
            print(f"    Max:    {max_val}")
            print(f"    Spread across metric means: {spread}")
            temp_df = df[target_columns].copy()
            temp_df['Expert'] = expert_id
            temp_df['AnonLevel'] = f"A{['None', 'Anon1', 'Anon2'].index(anon_level)}"
            all_boxplot_frames.append(temp_df)
    # Boxplot across all experts and AnonLevels
    if all_boxplot_frames:
        full_df = pd.concat(all_boxplot_frames, ignore_index=True)

        box_data = []
        for (expert, level), group_df in full_df.groupby(['Expert', 'AnonLevel']):
            score = group_df[['Grade', 'QU1', 'QU2', 'QU3']].values.flatten()
            for val in score:
                box_data.append({'Expert': expert, 'AnonLevel': level, 'Score': val})

        box_df = pd.DataFrame(box_data)
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=box_df, x='AnonLevel', y='Score', hue='Expert')
        plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
        plt.xlabel('Anonymization Level')
        plt.ylabel('Score')
        plt.title('Aggregated EM Score Distribution per Expert and AnonLevel')
        plt.grid(True, axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()