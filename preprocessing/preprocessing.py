import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer

# Load datasets
ds_ag = load_dataset('wangrongsheng/ag_news')
ds_20 = load_dataset("SetFit/20_newsgroups")

# Combine train and test splits
df_ag = pd.concat([ds_ag['train'].to_pandas(), ds_ag['test'].to_pandas()],
               ignore_index=True)
df_20 = pd.concat([ds_20['train'].to_pandas(), ds_20['test'].to_pandas()],
               ignore_index=True)

# Check for NaN or empty entries in AG News
mask_ag = df_ag['text'].isna() | (df_ag['text'].fillna('').str.strip().str.len() == 0)
print(f"AG News empty rows: {mask_ag.sum()}") # no empty rows

# Check for NaN or empty entries in Twenty Newsgroups
mask_empty = df_20['text'].isna() | (df_20['text'].fillna('').str.strip().str.len() == 0)
print(f"Total empty (NaN or whitespace-only): {mask_empty.sum()}") # 515 empty rows

print(df_20[mask_empty]['label_text'].value_counts())
print("\nFraction empty per category:")
print((df_20[mask_empty]['label_text'].value_counts() 
       / df_20['label_text'].value_counts()).sort_values(ascending=False))

# Drop rows with NaN or empty entries
print(f"Before: {len(df_20)}")
df_20 = df_20.dropna(subset=['text'])
df_20 = df_20[df_20['text'].str.strip() != '']
df_20 = df_20.reset_index(drop=True)
print(f"After: {len(df_20)}")

# Add label text for AG News
ag_labels = {
    0: 'world',
    1: 'sports',
    2: 'business',
    3: 'sci_tech',
}

df_ag['label_text'] = df_ag['label'].map(ag_labels)

# Add token counts
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

ag_lengths = []
twenty_lengths = []

for text in df_ag['text']:
    ag_lengths.append(len(tokenizer.encode(str(text), add_special_tokens=True, truncation=False)))
df_ag['minilm_tokens'] = ag_lengths

for text in df_20['text']:
    twenty_lengths.append(len(tokenizer.encode(str(text), add_special_tokens=True, truncation=False)))
df_20['minilm_tokens'] = twenty_lengths

# Save cleaned data
df_ag.to_csv('data/ag_news.csv', index=False)
df_20.to_csv('data/twenty_newsgroups.csv', index=False)
