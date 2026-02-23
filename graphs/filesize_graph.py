import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb


data = pd.read_csv("Labels_Sizes.csv")
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

data["Family"] = data["Class"].map(families)

average = data.groupby('Family')[['asm', 'bytes']].agg(['mean', 'min', 'max']).reset_index()

print(average.columns)

average.columns = ['_'.join(col).strip('_') for col in average.columns.values]


plt.figure(figsize=(12, 7))

plt.vlines(x=average['Family'], ymin=average['asm_min'], ymax=average['asm_max'], color='gray')
plt.scatter(average['Family'], average['asm_min'], color='blue', label='Min Size',zorder=3)
plt.scatter(average['Family'], average['asm_max'], color='red', label='Max Size',zorder=3)
plt.scatter(average['Family'], average['asm_mean'], color='black', marker='X', label='Mean',zorder=4)

# Formatting
plt.yscale('log') # Essential due to large size differences
plt.title('Sizes for .asm files')
plt.ylabel('Size in Bytes (Log Scale)')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=1)

plt.tight_layout()
plt.savefig("ASM Filesizes")

plt.figure(figsize=(12, 7))

plt.vlines(x=average['Family'], ymin=average['bytes_min'], ymax=average['asm_max'], color='gray')
plt.scatter(average['Family'], average['bytes_min'], color='blue', label='Min Size',zorder=3)
plt.scatter(average['Family'], average['bytes_max'], color='red', label='Max Size',zorder=3)
plt.scatter(average['Family'], average['bytes_mean'], color='black', marker='X', label='Mean',zorder=4)

# Formatting
plt.yscale('log') # Essential due to large size differences
plt.title('Sizes for .bytes files')
plt.ylabel('Size in Bytes (Log Scale)')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=1)

plt.savefig("BYTES Filesizes")

plt.show()