import pandas as pd
import numpy as np

df = pd.read_csv('data/processed/negative_data_batch_1.csv')

print('='*60)
print('DIAGNOSTIC: Outlier Score Analysis')
print('='*60)

# Find rows with extremely high scores
outliers = df[df['dms_score'] > 1000]
print(f'\n📊 Rows with score > 1000: {len(outliers):,}')

if len(outliers) > 0:
    # Extract DMS IDs - try matching against reference
    try:
        ref_df = pd.read_csv('../ProteinGym/reference_files/DMS_substitutions.csv')
        outlier_dms_ids = set()
        
        for seq_id in outliers['sequence_id'].unique():
            # Try matching against known DMS_ids
            for dms_id in ref_df['DMS_id'].values:
                if seq_id.startswith(dms_id + '_'):
                    outlier_dms_ids.add(dms_id)
                    break
        
        print(f'\n🔍 Affected DMS Assays ({len(outlier_dms_ids)}):')
        for dms_id in sorted(outlier_dms_ids):
            matching = outliers[outliers['sequence_id'].str.startswith(dms_id)]
            if len(matching) > 0:
                dms_info = ref_df[ref_df['DMS_id'] == dms_id].iloc[0]
                print(f'\n   - {dms_id}:')
                print(f'       Rows: {len(matching):,}')
                print(f'       Score range: [{matching["dms_score"].min():.2f}, {matching["dms_score"].max():.2f}]')
                print(f'       Mean: {matching["dms_score"].mean():.2f}')
                print(f'       Source: {dms_info["first_author"]} {dms_info["year"]}')
                print(f'       Selection type: {dms_info.get("coarse_selection_type", "N/A")}')
    except Exception as e:
        print(f'   Error matching: {e}')

# Score distribution
print(f'\n📈 Score Distribution:')
for p in [0, 25, 50, 75, 95, 99, 100]:
    val = df['dms_score'].quantile(p/100)
    print(f'   {p}th percentile: {val:.4f}')

# Check score ranges
print(f'\n🔬 Score Range Breakdown:')
print(f'   Score < -10: {(df["dms_score"] < -10).sum():,}')
print(f'   -10 <= Score < 0: {((df["dms_score"] >= -10) & (df["dms_score"] < 0)).sum():,}')
print(f'   0 <= Score < 10: {((df["dms_score"] >= 0) & (df["dms_score"] < 10)).sum():,}')
print(f'   10 <= Score < 100: {((df["dms_score"] >= 10) & (df["dms_score"] < 100)).sum():,}')
print(f'   100 <= Score < 1000: {((df["dms_score"] >= 100) & (df["dms_score"] < 1000)).sum():,}')
print(f'   Score >= 1000: {(df["dms_score"] >= 1000).sum():,}')

# Find max score row
max_score_row = df.loc[df['dms_score'].idxmax()]
print(f'\n🎯 Row with Maximum Score:')
print(f'   sequence_id: {max_score_row["sequence_id"]}')
print(f'   dms_score: {max_score_row["dms_score"]:.2f}')

