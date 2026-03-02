import numpy as np
from PIL import Image
import os
import math
from tqdm import tqdm
from multiprocessing import Pool

path = './train'
destination = './asm_images'
resize = (380,380)
errors = []
def asm_parser(filename, errors = []):

    filepath = os.path.join(path, filename) #cream filepathul cu numele fisierului inclusiv
    destpath = os.path.join(destination, filename) #cream filepathul cu numele fisierului pentru destinatie
    
    try:
        with open(filepath, 'rb') as f:
            hex_value = f.read()

        byte_array = np.frombuffer(hex_value, dtype=np.uint8)
        length = len(byte_array)
        width = int(math.sqrt(length))
        square_data = byte_array[:width*width].reshape((width, width))
        img = Image.fromarray(square_data)
        img = img.resize(resize, Image.LANCZOS)
        img = np.array(img)
        img = Image.fromarray(img)
        img.save(os.path.join(destination, filename.replace(".asm", ".png")))

    except Exception as e:
        errors.append(filename)
        return f"Error on {filename} : {e}"


if __name__ == "__main__":
    if not os.path.exists(destination):
        os.makedirs(destination)
    
    files = [f for f in os.listdir(path) if f.endswith('.asm')]
    total_f = len(files)
    print (f"Processing {total_f} files using all cores...")

    with Pool() as pool:
        list(tqdm(pool.imap_unordered(asm_parser, files), total = total_f, desc="Creating images", unit="file", colour="green", leave=True))

    print(f"The destination folder has received {len(os.listdir(destination))} files out of {len(files)} number of files.")
    print("Script complete!")
    print(errors)
    
