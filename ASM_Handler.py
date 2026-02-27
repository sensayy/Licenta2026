import os
import pandas as pd
from collections import Counter
from multiprocessing import Pool
from tqdm import tqdm

#https://www.researchgate.net/publication/228694812_Opcodes_as_predictor_for_malware
vocabulary = {'mov', 'push', 'call', 'pop', 'cmp', 'jz', 'lea', 'test', 'jmp', 'add', 'jnz', 'retn', 'xor', 'sbb', 'nop', 'imul', 'int'} #set in loc de lista ca sa fie mai rapid
asm_files = "./train" 


#Cod refolosit de la bytescsv.py si adaptat
def count_asm(filename):
    filepath = os.path.join(asm_files, filename)
    id = filename.split('.')[0] #scoatem id-ul fisierului ca sa stim cum il punem in csv

    if not filename.endswith(".asm"):
        return "Skipping .bytes file."

    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            code = f.read().split() #citim fisierul si il imparitm din nou intr-o lista cu fiecare pereche
        
        count = Counter(words for words in code if words in vocabulary) #folosim counter care returneaza practic un dictionar pentru fiecare valoare si numarul de aparitii
                                                                        #Cauta "cuvinte" din set din "codul" citit din fisier
        row = {'Id': id} #creem un dictionar nou pe care il vom transforma in csv
        for word in vocabulary: #pentru fiecare cuvant din dictionar (hexcode) adaugam in dictionar de cate ori a aparut
            row[word] = count.get(word, 0) 
        return row
    except Exception as e:
        print(f"Error has occured on  file {filename} : {e}")
        return None
    

    #Cod refolosit de la bytescsv.py
if __name__ == "__main__":
    asm = [f for f in os.listdir(asm_files) if f.endswith(".asm")] #accesam fisierele asm

    with Pool() as pool: #Initiem pool pentru multiprocesare
        results = list(tqdm(pool.imap_unordered(count_asm, asm), total = len(asm), desc = "Counting opcode...", unit = "file", colour = "green", leave = True))

    results = [r for r in results if r is not None]

    features_df = pd.DataFrame(results)
    labels_df = pd.read_csv("ByteCount_Size_Labels.csv")

    final_df = pd.merge(labels_df, features_df, on="Id")
    final_df.to_csv("Full_Features.csv", index = False)
    print(f"Done! Final file contains {final_df.shape[0]} rows and {final_df.shape[1]} columns.")