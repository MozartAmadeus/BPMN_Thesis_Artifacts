import os

import numpy as np
import pandas as pd
from typing import Dict, List

#ToDo Wir brauchen auch noch irgendwas veryfy data. schaut dass alle em om pdm namen matchen und zeigt wenn nicht. Zeicht auch nan values
from sklearn.preprocessing import StandardScaler, \
    MinMaxScaler


def load_pdm(pdm_path: str) -> pd.DataFrame:
    """Load process description metrics (PDM) with column cleaning and type conversion."""
    df_pdm = pd.read_csv(pdm_path, sep=';')
    df_pdm.columns = [col.strip().replace(' ', '_') for col in df_pdm.columns]

    if df_pdm.columns[0].lower() != 'name':
        raise ValueError("First column should be 'Name' for process identifier")

    df_pdm.rename(columns={df_pdm.columns[0]: 'process_id'}, inplace=True)
    df_pdm['process_id'] = df_pdm['process_id'].str.lower()
    df_pdm.fillna(0.0, inplace=True)

    float_cols = [
        'HasMissingTasks', 'Syllables', 'Content_Words', 'Sentences',
        'Irrelevant_Information', 'Extraneous_Words', 'Words'
    ]
    for col in float_cols:
        if col in df_pdm.columns:
            df_pdm[col] = df_pdm[col].astype(float)

    # Invert 'Correctness' column (1=best → 5=worst) to (1=worst → 5=best)
    if 'Correctness' in df_pdm.columns:
        df_pdm['Correctness'] = 6 - df_pdm['Correctness']

    
    # Normalize numeric columns (exclude boolean and process_id)
    numeric_cols = df_pdm.select_dtypes(include=[np.number]).columns.tolist()
    bool_cols = [
        col for col in numeric_cols
        if set(df_pdm[col].dropna().unique()) <= {0, 1}
           and col not in {'process_id', 'PDMAND_(Split)', 'PDMAND_(Join)'}
    ]
    
    norm_cols = [col for col in numeric_cols if col not in bool_cols and col != 'process_id']
    scaler = StandardScaler()
    df_pdm[norm_cols] = scaler.fit_transform(df_pdm[norm_cols])

    return df_pdm


def convert_decimal_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Convert comma-decimal object columns to float where possible, excluding 'process_id'"""
    for col in df.columns:
        if df[col].dtype == 'object' and col.lower() != 'process_id':
            try:
                df[col] = df[col].str.replace(',', '.').astype(float)
            except Exception:
                continue
    return df


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize and rename 'Name' to 'process_id' if present"""
    df.columns = [col.strip().replace(' ', '_') for col in df.columns]
    if 'Name' in df.columns:
        df.rename(columns={'Name': 'process_id'}, inplace=True)
    return df


def load_om(root_dir: str) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Load Output Metrics (OM) files into nested dict [llm][subgroup] = DataFrame"""
    om_data = {}
    for llm in os.listdir(root_dir):
        llm_path = os.path.join(root_dir, llm)
        if not os.path.isdir(llm_path):
            continue
        om_data[llm] = {}
        for subgroup in os.listdir(llm_path):
            file_path = os.path.join(llm_path, subgroup, f"OutputMetrics_{llm}_{subgroup}.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, sep=';')
                df = standardize_columns(df)
                df = convert_decimal_strings(df)

                # Ensure process_id is standardized
                if 'process_id' in df.columns:
                    df['process_id'] = df['process_id'].str.lower()
                else:
                    print(f"❌ 'process_id' column missing after standardization: {file_path}, skipping.")
                    continue

                # Rename and standardize error column
                if 'HatFehler' in df.columns:
                    df.rename(columns={'HatFehler': 'HasErrors'}, inplace=True)
                if 'HasErrors' in df.columns:
                    df['HasErrors'] = df['HasErrors'].astype(float)

                # Normalize Group column
                if 'Group' in df.columns:
                    df['Group'] = df['Group'].replace('Meins', 'Prototype')
                else:
                    df['Group'] = llm  # fallback if Group not in file

                # Extract AnonLevel from subgroup name (e.g., A0 → None, A1 → Anon1)
                try:
                    numeric_level = int(subgroup[1])  # subgroup is like 'A0', 'A1', ...
                    level_label = ['None', 'Anon1', 'Anon2'][numeric_level]
                except Exception as e:
                    level_label = 'Unknown'
                    print(f"⚠️ Could not parse AnonLevel from: {subgroup}")

                df['AnonLevel'] = level_label

                df = df.fillna(0)  # Replace NaNs with 0 globally

                # One-hot encode Group and AnonLevel
                df = pd.get_dummies(df, columns=['Group', 'AnonLevel'], dtype=int)

                # Ensure all possible dummy columns exist (padding missing with 0)
                for col in ['Group_Aalst', 'Group_Aau', 'Group_Prototype', 'AnonLevel_None', 'AnonLevel_Anon1', 'AnonLevel_Anon2']:
                    if col not in df.columns:
                        df[col] = 0

                # Normalize numeric columns (exclude boolean and process_id)
                numeric_cols = df.select_dtypes(include='number').columns.tolist()
                exclude = {
                    'process_id', 'HasErrors', 'Group_Prototype', 'AnonLevel_None',
                    'Group_Aalst', 'Group_Aau', 'AnonLevel_Anon1', 'AnonLevel_Anon2'
                }
                norm_cols = [col for col in numeric_cols if col not in exclude]
                scaler = StandardScaler()
                df[norm_cols] = scaler.fit_transform(df[norm_cols])

                om_data[llm][subgroup] = df

    return om_data


# add process id und keys!!!
def load_em(root_dir: str) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Load Expert Metrics (EM)
    """
    grouped_em_data = {}

    for expert_id in os.listdir(root_dir):
        expert_path = os.path.join(root_dir, expert_id)
        if not os.path.isdir(expert_path):
            continue

        if expert_id not in grouped_em_data:
            grouped_em_data[expert_id] = {}

        for llm in os.listdir(expert_path):
            llm_path = os.path.join(expert_path, llm)
            if not os.path.isdir(llm_path):
                continue

            for anon_level in os.listdir(llm_path):
                anon_path = os.path.join(llm_path, anon_level)
                if not os.path.isdir(anon_path):
                    continue

                file_path = os.path.join(anon_path, f"ExpertMetrics_{expert_id}_{llm}_{anon_level}.csv")
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path, sep=';')
                    df = standardize_columns(df)
                    df = convert_decimal_strings(df)

                    if 'process_id' in df.columns:
                        df['process_id'] = df['process_id'].str.lower()
                    else:
                        print(f"❌ 'process_id' column missing after standardization: {file_path}, skipping.")
                        continue

                    # Replace NaN in relevant expert columns with 1
                    for col in ['ThrownOut', 'Grade', 'QU1', 'QU2', 'QU3']:
                        if col in df.columns:
                            df[col] = df[col].fillna(1)

                    # Cast boolean columns to float for consistency
                    if 'HasErrors' in df.columns:
                        df['HasErrors'] = df['HasErrors'].astype(float)
                    if 'ThrownOut' in df.columns:
                        df['ThrownOut'] = df['ThrownOut'].astype(float)

                    # Inject metadata columns
                    df['Group'] = llm
                    numeric_level = int(anon_level[1])  # 'A0' -> 0
                    level_label = ['None', 'Anon1', 'Anon2'][numeric_level]  # map 0 → None, etc.
                    df['AnonLevel'] = level_label

                    #df['Expert'] = expert_id --> we do not need column expert

                    # One-hot encode 'Group' and 'AnonLevel'
                    df = pd.get_dummies(df, columns=['Group', 'AnonLevel'], dtype=int)

                    # Ensure all possible dummy columns exist (padding missing with 0)
                    for col in ['Group_Aalst', 'Group_Aau', 'Group_Prototype', 'AnonLevel_None', 'AnonLevel_Anon1', 'AnonLevel_Anon2']:
                        if col not in df.columns:
                            df[col] = 0

                    if level_label not in grouped_em_data[expert_id]:
                        grouped_em_data[expert_id][level_label] = []
                    grouped_em_data[expert_id][level_label].append(df)

    return {
        expert_id: {
            level: pd.concat(frames, ignore_index=True)
            for level, frames in level_dict.items()
        }
        for expert_id, level_dict in grouped_em_data.items()
    }



def combine_experts(em: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, pd.DataFrame]:
    """Combine expert-level EM data into a single dataframe per AnonLevel by averaging matching process_id + one-hot columns."""
    from collections import defaultdict

    combined_by_anon: Dict[str, List[pd.DataFrame]] = defaultdict(list)

    for expert_id, anon_dict in em.items():
        for anon_level, df in anon_dict.items():
            # If it's a DataFrame (not a list), wrap it in a list for uniformity
            if isinstance(df, pd.DataFrame):
                combined_by_anon[anon_level].append(df)
            elif isinstance(df, list):
                for sub_df in df:
                    if isinstance(sub_df, pd.DataFrame):
                        combined_by_anon[anon_level].append(sub_df)

    result: Dict[str, pd.DataFrame] = {}
    scalers: Dict[str, MinMaxScaler] = {}
    target_columns = ['Grade', 'QU1', 'QU2', 'QU3']

    for anon_level, df_list in combined_by_anon.items():
        combined_df = pd.concat(df_list, axis=0, ignore_index=True)

        # Identify columns to group by (process_id + one-hot columns)
        group_cols = ['process_id'] + [col for col in combined_df.columns if col.startswith('Group_') or col.startswith('AnonLevel_')]

        # Group and average
        averaged = combined_df.groupby(group_cols, as_index=False)[target_columns].mean()
        '''
        scaler = MinMaxScaler()
        #scaler = StandardScaler()
        averaged[target_columns] = scaler.fit_transform(averaged[target_columns])
        scalers[anon_level] = scaler
        '''
        result[anon_level] = averaged

    return result

def extract_single_hot_label(row: pd.Series, prefix: str) -> str:
    for col in row.index:
        if col.startswith(prefix) and row[col] == 1:
            return col.replace(prefix, '')
    return 'UNKNOWN'

def create_combined_key_em_combined(em_combined: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Add a 'combined_id' column to each DataFrame in em_combined by combining process_id, Group, and AnonLevel.
    This will uniquely identify each sample across different LLM groups and anonymization levels.
    """
    updated = {}
    for anon_level, df in em_combined.items():
        df = df.copy()

        df['AnonLevel'] = df.apply(lambda row: extract_single_hot_label(row, 'AnonLevel_'), axis=1)
        df['Group'] = df.apply(lambda row: extract_single_hot_label(row, 'Group_'), axis=1)

        df['combined_id'] = df['process_id'].str.lower() + '_' + df['AnonLevel'] + '_' + df['Group']
        df.set_index('combined_id', inplace=True)

        updated[anon_level] = df

    return updated


def create_combined_key_om(om_data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, pd.DataFrame]:
    """Flatten and merge OM data into a single DataFrame per AnonLevel with combined key."""
    from collections import defaultdict

    merged: Dict[str, List[pd.DataFrame]] = defaultdict(list)
    subgroup_map = {'A0': 'None', 'A1': 'Anon1', 'A2': 'Anon2'}

    for llm, subgroups in om_data.items():
        for subgroup, df in subgroups.items():
            group_cols = [col for col in df.columns if col.startswith('Group_')]
            anon_cols = [col for col in df.columns if col.startswith('AnonLevel_')]

            df = df.copy()

            group_val = df[group_cols].idxmax(axis=1).str.replace('Group_', '', regex=False)
            anon_val = df[anon_cols].idxmax(axis=1).str.replace('AnonLevel_', '', regex=False)

            df['combined_id'] = df['process_id'].str.lower() + '_' + anon_val + '_' + group_val
            df.set_index('combined_id', inplace=True)

            level_label = subgroup_map.get(subgroup, subgroup)
            merged[level_label].append(df)

    result: Dict[str, pd.DataFrame] = {}
    for anon_level, df_list in merged.items():
        result[anon_level] = pd.concat(df_list, axis=0, ignore_index=False)

    return result











def merge_pdm_with_om(pdm: pd.DataFrame, om_df: pd.DataFrame) -> pd.DataFrame:
    """Merge PDM with one OM DataFrame on process_id (case-insensitive), returning a new DataFrame.
    Raises warnings if duplicate process_id entries are found or if rows are dropped during merge.
    """
    pdm = pdm.copy()
    om_df = om_df.copy()

    pdm['process_id'] = pdm['process_id'].str.lower()
    om_df['process_id'] = om_df['process_id'].str.lower()

    if pdm['process_id'].duplicated().any():
        print("⚠️ Warning: Duplicate process_id values in PDM")
    if om_df['process_id'].duplicated().any():
        print("⚠️ Warning: Duplicate process_id values in OM")

    merged_df = om_df.merge(pdm, on="process_id", how="inner")

    if len(merged_df) < len(om_df):
        print(f"⚠️ Warning: {len(om_df) - len(merged_df)} rows dropped during merge (unmatched process_id)")

    return merged_df