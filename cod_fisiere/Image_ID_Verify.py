import os
import pandas as pd

# Check what CSV looks like
df = pd.read_csv("trainLabels.csv")
print(df.head())
print(f"\nCSV Id example: '{df['Id'].iloc[0]}'")

# Check what files actually exist in  image folder
image_folder = "images/" 
existing_files = set(os.path.splitext(f)[0] for f in os.listdir(image_folder))

missing = []
for fname in df["Id"]:
    if fname not in existing_files:
        missing.append(fname)

print(f"Missing: {len(missing)} / {len(df)}")
if missing:
    print(f"Example missing: {missing[:3]}")
    print(f"Example existing: {list(existing_files)[:3]}")
# Check if lowercase versions match
for fname in missing[:3]:
    if fname.lower() in {f.lower() for f in existing_files}:
        print(f"Case mismatch: CSV has '{fname}'")
        # Find the actual filename
        actual = next(f for f in existing_files if f.lower() == fname.lower())
        print(f"  Actual file: '{actual}'")