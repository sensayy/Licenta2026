import os

files = os.listdir("./images")
for f in files:
    rename = f.replace(".bytes.png", ".png")
    old_path = os.path.join("./images", f)
    new_path = os.path.join("./images", rename)

    os.rename(old_path, new_path)

    print(f"Renamed file {f} to {rename}")

