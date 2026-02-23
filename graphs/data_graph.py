import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

categories = pd.read_csv("trainLabels.csv")

families = {
    1: "Ramnit",
    2: "Lollipop",
    3: "Kelihos_ver3",
    4: "Vundo",
    5: "Simda",
    6: "Tracur",
    7: "Kelihos_ver1",
    8: "Obfuscator.ACY",
    9: "Gatak"
}


categories["Family"] = categories["Class"].map(families)
counts = categories["Family"].value_counts()

plt.figure(figsize=(12,7))
plot = sb.countplot(x = "Family", data = categories, hue="Family", legend=True)
for bar in plot.containers:
    plot.bar_label(bar, padding = 3)
plt.title("Distribution of Train Dataset Malware Classes")
plt.xlabel("Malware Name")
plt.ylabel("Count of Appereances in Dataset")
plt.savefig("Train Dataset Analysis Count")

plt.figure(figsize=(12,7))
plt.pie(counts, labels =counts.index, autopct='%1.1f%%')

plt.savefig("Train Dataset Analysis Piechart")
plt.show()


