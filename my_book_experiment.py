import zipfile
import io
import requests
import os

from requests.models import DecodeError

from elliot.run import run_experiment

url = "http://www2.informatik.uni-freiburg.de/~cziegler/BX/BX-CSV-Dump.zip"
print(f"Getting BookCrossing from : {url} ..")
response = requests.get(url)

bookcrossing_ratings = []

print("Extracting ratings ..")
with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
    for line in zip_ref.open("BX-Book-Ratings.csv"):
        cleaned_line = "\t".join(str(line, "utf-8", errors = "replace").rstrip().replace("\"", "").split(";"))
        
        bookcrossing_ratings.append(cleaned_line + "\n")
    bookcrossing_ratings.pop(0)

print("Printing ratings.tsv to data/bookcrossing/ ..")
os.makedirs("data/bookcrossing", exist_ok=True)
with open("data/bookcrossing/dataset.tsv", "w", encoding = "utf-8") as f:
    f.writelines(bookcrossing_ratings, )

print("Done! We are now starting the Elliot's experiment")
run_experiment("config_files/my_book_config.yml")