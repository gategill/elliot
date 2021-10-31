import zipfile
import io
import requests
import os

from elliot.run import run_experiment

#url = "http://files.grouplens.org/datasets/hetrec2011/hetrec2011-movielens-2k-v2.zip"
#print(f"Getting Movielens 2K from : {url} ..")
#response = requests.get(url)
#
#ml_2k_ratings = []
#
#print("Extracting ratings ..")
#with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
#    for line in zip_ref.open("user_ratedmovies.dat"):
#        first_three_cols = "\t".join(str(line, "utf-8").strip().split("\t")[:3])
#        ml_2k_ratings.append(first_three_cols + "\n")
#    ml_2k_ratings.pop(0)
#
#print("Printing ratings.tsv to data/movielens_2k/ ..")
#os.makedirs("data/movielens_2k", exist_ok=True)
#with open("data/movielens_2k/dataset.tsv", "w") as f:
#    f.writelines(ml_2k_ratings)
#
print("Done! We are now starting the Elliot's experiment")
run_experiment("config_files/my_movie_config.yml")