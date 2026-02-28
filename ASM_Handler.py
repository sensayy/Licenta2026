import os
import pandas as pd
from collections import Counter
from multiprocessing import Pool
from tqdm import tqdm

''' 
    Creez mai multe vocabulare de opcode-uri bazate pe mai multe cercetari din domeniu
    pentru a vedea care set este optim in antrenarea modelelor de machine learning pentru detectia de malware.
'''



#https://www.researchgate.net/publication/228694812_Opcodes_as_predictor_for_malware
#Bilar, D. (2007). Opcodes as predictor for malware. International Journal of Electronic Security and Digital Forensics.
vocabulary1 = {'mov', 'push', 'call', 'pop', 'cmp', 'jz', 'lea', 'test', 'jmp', 'add', 'jnz', 'retn', 'xor', 'sbb', 'nop', 'imul', 'int'} #set in loc de lista ca sa fie mai rapid

#Santos, I., Brezo, F., Ugarte-Pedrero, X., & Bringas, P. G. (2013). Opcode sequences as representation of executables for data-mining-based unknown malware detection. Information Sciences.
#Rad, B. B., Masrom, M., & Ibrahim, S. (2012). Opcodes histogram for classifying metamorphic portable executables malware. ECIW.
vocabulary2 = {
    'mov', 'push', 'call', 'pop', 'cmp', 'jz', 'lea', 'test', 'jmp', 'add',
    'jnz', 'retn', 'xor', 'sbb', 'nop', 'imul', 'int',
    'sub', 'and', 'or', 'inc', 'dec', 'movzx', 'movsx', 'sar', 'shr', 'shl',
    'mul', 'div', 'neg', 'not', 'je', 'jne', 'jl', 'jle', 'jg', 'jge',
    'jb', 'jbe', 'ja', 'jae', 'ret', 'leave', 'enter'
}

#Roundy, K. A., & Miller, B. P. (2013). Binary-code obfuscations in prevalent packer tools. ACM Computing Surveys. — supports the packer/unpacking loop opcodes (pushad, popad, rep, stos etc.)
#Ferrie, P. (2008). Attacks on virtual machines. Symantec. — supports anti-VM opcodes (cpuid, rdtsc, in, out)
vocabulary3 = {
    'xor', 'ror', 'rol', 'shl', 'shr', 'not', 'and', 'or',          # crypto/obfuscation
    'call', 'retn', 'ret', 'int', 'jmp',                              # control flow
    'push', 'pop', 'mov', 'lea',                                      # data movement
    'in', 'out', 'cpuid', 'rdtsc',                                    # anti-VM/timing
    'pushad', 'popad', 'pushfd', 'popfd',                             # context saving (packers)
    'rep', 'repe', 'repne', 'movs', 'stos', 'lods', 'scas',          # memory operations (unpacking loops)
    'nop', 'hlt'                                                       # evasion padding
}

#Moskovitch, R., et al. (2008). Unknown malcode detection using OPCODE representation. EuroISI. — supports using a small focused opcode set
#Karim, M. E., et al. (2005). Malware phylogeny generation using permutations of code. Journal in Computer Virology. — supports bitwise ops as family discriminators
vocabulary4 = {
    'xor', 'shl', 'shr', 'ror', 'rol',     # bitwise — high in crypto malware
    'call', 'retn', 'jmp', 'jz', 'jnz',    # control flow skeleton
    'push', 'pop', 'mov',                   # stack/memory essentials
    'int', 'in', 'out',                     # privileged/system interaction
    'rdtsc', 'cpuid',                       # anti-analysis
    'nop', 'imul'                           # packing indicators
}

vocabularies = [vocabulary1, vocabulary2, vocabulary3, vocabulary4]
asm_files = "./train" 


#Cod refolosit de la bytescsv.py si adaptat
def count_asm(filename):
    filepath = os.path.join(asm_files, filename)
    id = filename.split('.')[0] #scoatem id-ul fisierului ca sa stim cum il punem in csv

    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            code = f.read().split() #citim fisierul si il imparitm din nou intr-o lista cu fiecare pereche

        rows = []
        for vocab in vocabularies:
            count = Counter(word for word in code if word in vocab) #numaram de cate ori apare fiecare opcode din vocabular in codul sursa, folosind Counter pentru a crea un dictionar cu opcode-urile ca chei si numarul de aparitii ca valori
            row = {'Id': id} #pentru fiecare vocabular, creeaza un dictionar care contine fiecare opcode din vocabular si numarul sau de aparitii in cod (sau 0 daca nu apare)
            for word in vocab:
                row[word] = count.get(word, 0) #daca opcode-ul nu apare in cod, count.get(word, 0) va returna 0
            rows.append(row)
        return rows
    except Exception as e:
        print(f"Error has occured on  file {filename} : {e}")
        return None
    
    #Cod refolosit de la bytescsv.py
if __name__ == "__main__":

    asm = [f for f in os.listdir(asm_files) if f.endswith(".asm")] #accesam fisierele asm

    with Pool() as pool: #Initiem pool pentru multiprocesare
        results = list(tqdm(pool.imap_unordered(count_asm, asm), total = len(asm), desc = "Counting opcode...", unit = "file", colour = "green", leave = True))

    results = [r for r in results if r is not None]
    results_per_vocab = [[file_rows[i] for file_rows in results] for i in range(len(vocabularies))] #rezultatele sunt o lista de liste de dictionare, 
                                                            # unde fiecare sublista corespunde unui vocabular si contine un dictionar pentru fiecare fisier asm cu numarul de aparitii al fiecarui 
                                                            # opcode din vocabularul respectiv

    labels_df = pd.read_csv("ByteCount_Size_Labels.csv")

    for i, vocab_results in enumerate(results_per_vocab, 1):
        features_df = pd.DataFrame(vocab_results)
        final_df = pd.merge(labels_df, features_df, on="Id")
        out_name = f"Full_Features_vocab{i}.csv"
        final_df.to_csv(out_name, index=False)
        print(f"Vocab {i}: {final_df.shape[0]} rows, {final_df.shape[1]} columns -> {out_name}")