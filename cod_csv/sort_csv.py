# =============================================================================
# sort_csv.py
# Sorteaza CSV-ul pentru a se potrivi cu ordinea in care ImageFolder
# incarca imaginile: alfabetic dupa clasa (1-9), apoi alfabetic dupa
# numele fisierului (Id) in cadrul fiecarei clase.
#
# Ex: clasa 1 → [0AnoOZDNbPXIr2MRBSCJ, 0BzXkLm..., ...], clasa 2 → [...], etc.
#
# Rulare: python sort_csv.py
# Input:  csv/Full_Features_vocab2.csv
# Output: csv/Full_Features_vocab2_sorted.csv
# =============================================================================

import pandas as pd

INPUT_CSV  = "csv/Full_Features_vocab2.csv"
OUTPUT_CSV = "csv/Full_Features_vocab2_sorted.csv"

df = pd.read_csv(INPUT_CSV)
print(f"Loaded {len(df)} rows from {INPUT_CSV}")

# Sort by Class (1-9) then by Id (filename) alphabetically — exactly what ImageFolder does
df_sorted = df.sort_values(
    by=["Class", "Id"],
    ascending=[True, True],
    key=lambda col: col.astype(str)  # string sort, not numeric, to match ImageFolder behavior
).reset_index(drop=True)

df_sorted.to_csv(OUTPUT_CSV, index=False)
print(f"Sorted CSV saved to {OUTPUT_CSV}")

print(f"\nClass distribution:\n{df_sorted['Class'].value_counts().sort_index()}")
print(f"\nFirst 5 rows:\n{df_sorted[['Id', 'Class']].head()}")
print(f"\nLast 5 rows:\n{df_sorted[['Id', 'Class']].tail()}")