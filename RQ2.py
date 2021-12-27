import os
import argparse
import pandas as pd
from tqdm import tqdm
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from DGCNN import DGCNN

SEED = 42

parser = argparse.ArgumentParser(
    description="RQ 2 (Cross-architecture detection).")
parser.add_argument("--input", "-i",
                    nargs="?",
                    help="Target architecture.")
args = parser.parse_args()

collection = pd.read_csv('../IoT_Malware_Collection_2021.csv')
collection = pd.concat([collection, pd.read_csv(
    '../IoT_Benign_Collection_2021.csv')], ignore_index=True)
collection.fillna('benign', inplace=True)

train_paths, test_paths = [], []
for i, row in tqdm(collection.iterrows()):
    hash = row['SHA-256']
    dataset = row['Dataset']
    dataset = dataset.split(',')[0].replace('-', '_')
    if not os.path.exists(path := f'data/{dataset}/{hash}.txt'):
        continue
    if row['Architecture'] == args.input:
        test_paths.append(path)
    else:
        train_paths.append(path)
print(len(train_paths), len(test_paths))

model = DGCNN(
    train_paths=train_paths,
    test_paths=test_paths,
    report_path=f'reports/RQ2/{args.input}.txt'
)
model.run()
