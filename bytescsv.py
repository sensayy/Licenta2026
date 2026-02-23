import os
import pandas as pd
from collections import Counter
from multiprocessing import Pool
from tqdm import tqdm



hex = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d", "e", "f"]
vocabulary = [f"{i}{j}".upper() for i in hex for j in hex]
vocabulary.append("??")
##### Se creeaza un "vocabular" cu toate posibilitatile de combinatii pentru hex

parsed_bytes = "./parsed_bytes" 

def count_bytes(filename):
    filepath = os.path.join(parsed_bytes, filename)
    id = filename.split('.')[0] #scoatem id-ul fisierului ca sa stim cum il punem in csv

    try:
        with open(filepath, 'r') as f:
            hexcode = f.read().split() #citim fisierul si il imparitm din nou intr-o lista cu fiecare pereche
        count = Counter(hexcode) #folosim counter care returneaza practic un dictionar pentru fiecare valoare si numarul de aparitii

        row = {'Id': id} #creem un dictionar nou pe care il vom transforma in csv
        for word in vocabulary: #pentru fiecare cuvant din dictionar (hexcode) adaugam in dictionar de cate ori a aparut
            row[word] = count.get(word, 0) 
        return row
    except Exception as e:
        print(f"Error has occured on  file {filename} : {e}")
        return None


if __name__ == "__main__":
    bytes_files = [f for f in os.listdir(parsed_bytes) if f.endswith(".bytes")] #accesam fisierele parsate

    with Pool() as pool:
        results = list(tqdm(pool.imap_unordered(count_bytes, bytes_files), total = len(bytes_files), desc = "Counting...", unit = "file", colour = "green", leave = True))

    results = [r for r in results if r is not None]

    features_df = pd.DataFrame(results)
    labels_df = pd.read_csv("Labels_Sizes.csv")

    final_df = pd.merge(labels_df, features_df, on="Id")
    final_df.to_csv("ByteCount_Size_Labels.csv", index = False)
    print(f"Done! Final file contains {final_df.shape[0]} rows and {final_df.shape[1]} columns.")






