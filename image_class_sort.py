import os
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm

def sort_images_by_class(image_dir):
    # Creeaza un dictionar pentru a stoca clasele si imaginile corespunzatoare
    class_dict = {}
    df = pd.read_csv('trainLabels.csv')  # Încarcă  CSV

    # Parcurge toate fisierele din directorul de imagini
    for filename in os.listdir(image_dir):
        if filename.endswith('.png'):
            id = filename.split('.')[0]  # Extrage ID-ul din numele fisierului
        #Corespundem ID-ului din numele fisierului cu clasa din CSV
            class_num = df.loc[df['Id'] == id, 'Class'].values[0]

            if class_num not in class_dict: #Cautam daca exista in dictionar
                class_dict[class_num] = [] #Daca nu exista, cream o noua intrare pentru acea clasa

            class_dict[class_num].append(filename) #Adaugam numele fisierului la lista de imagini pentru acea clasa

    return class_dict





def img_mover(sorted_images):
    image_directory = 'images'

#Cream un folder pentru fiecare clasa si punem fisierele corespunzatoare
    for class_num, filenames in sorted_images.items():
        if not os.path.exists(str(class_num)):
            os.makedirs(str(class_num))  # Creeaza un folder pentru clasa daca nu exista

        for filename in filenames:
            src_path = os.path.join(image_directory, filename)  # Calea sursa a imaginii
            dst_path = os.path.join(str(class_num), filename)  # Calea destinatie a imaginii
            os.rename(src_path, dst_path)  # Muta imaginea in folderul corespunzator clasei

if __name__ == "__main__":
    image_directory = 'images'  # Directorul care contine imaginile
    sorted_images = sort_images_by_class(image_directory)
    with Pool() as pool:
        list(tqdm(pool.imap_unordered(img_mover, [sorted_images]), total=len(sorted_images), desc="Moving Images", unit="file", colour="green", leave=True))

