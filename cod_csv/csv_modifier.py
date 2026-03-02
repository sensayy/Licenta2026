import pandas as pd
import os
'''
df = pd.read_csv("csv/Full_Features_FullASM.csv")
df = df.drop(columns=['bytes', 'asm'])
if not os.path.exists('modified_csv'):
    os.makedirs('modified_csv')
df.to_csv('modified_csv/Full_Features_FullASM_modified.csv', index=False)
'''

'''
files = ["Full_Features_vocab1.csv", "Full_Features_vocab2.csv", "Full_Features_vocab3.csv", "Full_Features_vocab4.csv"]
path = "csv/"

# 1. Start with the first file
final_df = pd.read_csv(f"{path}{files[0]}")

# 2. Loop through the rest
for file in files[1:]:
    print(f"Processing {file}...")
    df_next = pd.read_csv(f"{path}{file}")
    
    # Identify columns in df_next that are ALREADY in final_df
    # We keep 'Id' and 'Class' for the merge, but drop other duplicates
    duplicate_cols = [col for col in df_next.columns if col in final_df.columns and col not in ["Id", "Class"]]
    
    # Drop those duplicates from the new file before merging
    df_next = df_next.drop(columns=duplicate_cols)
    
    # Now merge safely
    final_df = pd.merge(final_df, df_next, on=["Id", "Class"], how="inner")

# 3. Save
final_df.to_csv(f"{path}Full_Features_FullASM.csv", index=False)
print("Done! Combined file saved.")
'''
