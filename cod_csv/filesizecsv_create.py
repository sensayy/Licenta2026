import pandas as pd

data = pd.read_csv("filesizes.csv")
data2 = pd.read_csv("trainLabels.csv")

final_data = pd.merge(data, data2, on = "Id")

final_data.to_csv("Labels_Sizes.csv", index = False)

