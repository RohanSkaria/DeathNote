import pandas as pd

df = pd.read_csv('data/processed/negative_data_batch_1.csv')

print('='*60)
print('DATASET BALANCE ANALYSIS')
print('='*60)

print(f'\n📊 Current Dataset:')
print(f'   Total samples: {len(df):,}')
print(f'   Label distribution:')
label_counts = df['label'].value_counts()
for label, count in label_counts.items():
    print(f'      Label {label}: {count:,} ({count/len(df)*100:.2f}%)')

if (df['label'] == 0).all():
    print(f'\n⚠️  CRITICAL ISSUE: Dataset is 100% failures (label=0)!')
    print(f'   This will cause the model to always predict 0')
    print(f'   Need to extract POSITIVE samples (label=1) as well!')
    
    print(f'\n💡 SOLUTION:')
    print(f'   Use DMS_score_bin column from source files:')
    print(f'   - DMS_score_bin = 0 → label = 0 (negative/failure)')
    print(f'   - DMS_score_bin = 1 → label = 1 (positive/success)')
else:
    print(f'\n✅ Dataset has both positive and negative samples!')

