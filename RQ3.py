import os
import pandas as pd
from tqdm import tqdm
from glob import glob
from datetime import datetime
from sklearn.utils import shuffle

from DGCNN import DGCNN

SEED = 42

collection_path = '../IoT_Malware_Collection_2021.csv'
collection = pd.read_csv(collection_path)

paths = []
for i, row in tqdm(collection.iterrows()):
    hash = row['SHA-256']
    dataset = row['Dataset']
    dataset = dataset.split(',')[0].replace('-', '_')
    if not os.path.exists(path := f'data/{dataset}/{hash}.txt'):
        continue
    date = row['Date']
    date = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
    paths.append((path, date))

paths = sorted(paths, key=lambda e: e[1])
paths, _ = zip(*paths)
malware_paths = list(paths)

benign_paths = list(glob('data/benign/*'))
benign_paths = shuffle(benign_paths, random_state=SEED)

mal_split = int(len(malware_paths)*7/10)
beg_split = int(len(benign_paths)*7/10)
train_paths = malware_paths[:mal_split] + benign_paths[:beg_split]
test_paths = malware_paths[mal_split:] + benign_paths[beg_split:]

model = DGCNN(
    train_paths=train_paths,
    test_paths=test_paths,
    report_path=f'reports/RQ3.txt'
)
model.run()
