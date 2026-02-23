import py7zr as pz
import pandas as pd
from pathlib import Path

sizes = []

with pz.SevenZipFile('train.7z', mode='r') as arhiva:
    file_list = arhiva.list()
    sizes = []
    for file in file_list:

        if file.is_directory:
            continue
        filename = Path(file.filename).name

        namesplit = filename.split('.')
        id = namesplit[0]
        extension = namesplit[-1]  #-1 in caz ca id-ul contine mai multe puncte in afara de cel pentru extensie
        sizes.append({
                'Id': id,
                "Extension": extension,
                'Size' : file.uncompressed
            })

dataframe = pd.DataFrame(sizes)
sizeframe = dataframe.pivot(index = 'Id', columns = 'Extension', values = 'Size').reset_index()
sizeframe.to_csv('filesizes.csv', index = False)


print("All done!")