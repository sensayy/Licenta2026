import os
from multiprocessing import Pool
from tqdm import tqdm

def process_file(filename):

    path = "./train"
    destination = "./parsed_bytes"
    
    if not filename.endswith('bytes'): #In caz ca se citeste eronat un .asm file
        return "Skipped .asm file" 
    
    filepath = os.path.join(path, filename) #cream filepathul cu numele fisierului inclusiv
    destpath = os.path.join(destination, filename) #cream filepathul cu numele fisierului pentru destinatie
    
    try:
        no_address_rows = [] #lista goala in care incarcam liniile dupa ce a fost scoasa adresa
        with open(filepath, 'r') as f: # deschidem pentru citit
            for line in f: #pentru fiecare linie
                row = line.split() #imparte in lista de genu ['1', '2', '3']
                if len(row) > 1: 
                    no_address_rows.append(" ".join(row[1:])) #row[1:] scoate primul element care este si adresa
        
        with open(destpath, 'w') as f: #deschidem pentru scris la destinatie
            f.write("\n".join(no_address_rows)) #scriem fisierul la loc folosind vectorul creat mai sus care acum contine toate datele fisierului modificat
        return f"Processed file: {filename}" 
    except Exception as e:
        return f"Failed to process file : {filename}: {e}"




if __name__ == "__main__":

    path = "./train"
    destination = "./parsed_bytes"

    if not os.path.exists(destination): #se creeaza folderul destinatie daca nu exista
        os.makedirs(destination)
    
    files = [f for f in os.listdir(path) if f.endswith(".bytes")] # luam fiecare fisier care se termina cu .bytes
    with Pool() as pool: #Pool este de la multiprocessing care creaza un "Pool" de "Muncitori" care default este numarul de core-uri ale Pc-ului in cazul meu 20
        list(tqdm(pool.imap_unordered(process_file, files), total = len(files), desc="Removing Addresses", unit="file", colour="green", leave=True)) 
        
        ###List este un fel de wrapper pentru un loop, necesar pentru ca imap_unordered nu functioneaza daca nu este "activat" printr-un loop sau un wrapper
        ###tqdm este alt wrapper folosit pentru progress bar, ia mai multe argumente. In cazul asta primul este procesul care se ruleaza, un total sa stie sa faca bar-ul
        ###si restu sunt doar de aspect general. Leave inseamna practic sa nu stearga progress barul cand a ajuns la 100%
    
    
    print(f"The destination folder has received {len(os.listdir(destination))} files out of {len(files)} number of files.")
    print("Script complete!")

