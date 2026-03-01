import pandas as pd

df = pd.read_csv("ByteCount_Size_Labels.csv")
df = df.sort_values(by = "Class", ascending=True)

#cols = list(df.columns)
#cols.remove("Class")
#cols.append("Class")
#df = df[cols]


df.to_csv("ByteCount_Size_Labels.csv", index=False)

