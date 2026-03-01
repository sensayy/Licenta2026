import numpy as np
from PIL import Image
import os
import math
from tqdm import tqdm
from multiprocessing import Pool

path = './parsed_bytes'
destination = './images'
img_width = 256
errors = []
def hex_img(filename, errors = []):

    if not filename.endswith('bytes'): #In caz ca se citeste eronat un .asm file
        return "Skipped .asm file"
    
    filepath = os.path.join(path, filename) #cream filepathul cu numele fisierului inclusiv
    destpath = os.path.join(destination, filename) #cream filepathul cu numele fisierului pentru destinatie
    
    try:

        with open(filepath, 'r') as f:
            hex_value = f.read().split()

        byte_array = [int(h, 16) if h!= '??' else 0 for h in hex_value]
    
        byte_array = np.array(byte_array, dtype=np.uint8)

        height= len(byte_array) // img_width

        if height == 0:
            return f"Skipped {filename}: File too small"
        
        final_data = byte_array[:height * img_width]

        img_matrix = final_data.reshape((height, img_width))
        img = Image.fromarray(img_matrix, 'L')
        img.save(os.path.join(destination, filename + ".png"))

        
    
    except Exception as e:
        errors.append(filename)
        return f"Error on {filename} : {e}"

if __name__ == "__main__":
    if not os.path.exists(destination):
        os.makedirs(destination)
    
    files = [f for f in os.listdir(path) if f.endswith('.bytes')]
    total = len(files)
    print (f"Processing {total} files using all cores...")

    with Pool() as pool:
        list(tqdm(pool.imap_unordered(hex_img, files),total = len(files), desc="Removing Addresses", unit="file", colour="green", leave=True))

    print(f"The destination folder has received {len(os.listdir(destination))} files out of {len(files)} number of files.")
    print("Script complete!")
    print(errors)
    
