import zipfile
import io
from pandas.io.pytables import IndexCol
import requests
import os
import numpy as np
import pandas as pd

from elliot.run import run_experiment

if not os.path.exists("./data/jester/jester-data-2.xls"):
    url = "https://goldberg.berkeley.edu/jester-data/jester-data-2.zip"
    print(f"Getting Jester from : {url} ..")
    response = requests.get(url)
    
    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
        #for line in zip_ref.open("jester-data-2.xls"):
        zip_ref.extract("jester-data-2.xls","./data/jester/")

print("Extracting ratings ..")
jester_ratings = []
n = 0

xls = pd.read_excel("./data/jester/jester-data-2.xls", header = None, index_col = None)
xls = xls.drop(columns=[0])
#print(xls.iloc[1])

for i, column in enumerate(xls):
    print(xls[column])
    for j, row in enumerate(xls[column]):
        if row != 99:
            o = "{}\t{}\t{}\n".format(j + 1, i + 1, row)
        #print(row)
        break
    break
#sheet = xls.parse(0)
#print(sheet.iloc[0])
#    print(line)
#    #first_three_cols = "\t".join(str(line, "utf-8").strip().split("\t")[:3])
#    jester_ratings.append(line + "\n")
#    n += 1
#    if n == 5:
#        break
#    #jester_ratings.pop(0)


print("Printing ratings.tsv to data/jester_ratings/ ..")
os.makedirs("data/jester", exist_ok=True)
with open("data/jester/dataset.tsv", "w") as f:
    f.writelines(jester_ratings)

print("Done! We are now starting the Elliot's experiment")
#run_experiment("config_files/my_jester_config.yml")