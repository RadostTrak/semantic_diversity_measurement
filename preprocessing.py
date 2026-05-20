import pandas as pd

# Read in data
df_ag = pd.read_csv('data/ag_news.csv')
df_20 = pd.read_csv('data/twenty_newsgroups_raw.csv')

# Check for NaN or empty entries in AG News
mask_ag = df_ag['text'].isna() | (df_ag['text'].fillna('').str.strip().str.len() == 0)
print(f"AG News empty rows: {mask_ag.sum()}") # AG News has no empty rows

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

# Save cleaned data
df_20.to_csv('data/twenty_newsgroups.csv', index=False)
