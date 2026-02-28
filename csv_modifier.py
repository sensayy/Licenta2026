import pandas as pd
import os

df = pd.read_csv("Full_Features_vocab4.csv")
df = df.drop(columns=['bytes', 'asm'])
if not os.path.exists('modified_csv'):
    os.makedirs('modified_csv')
df.to_csv('modified_csv/Full_Features_vocab4_modified.csv', index=False)