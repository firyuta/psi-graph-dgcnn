import argparse
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from DGCNN import DGCNN

SEED = 42

parser = argparse.ArgumentParser(description="RQ 1 (Dataset-based detection).")
parser.add_argument("--input", "-i",
                    nargs="?",
                    help="Input dataset.")
args = parser.parse_args()

malware_paths = glob(f'data/{args.input}/*')
benign_paths = shuffle(glob('data/benign/*'),
                       random_state=SEED)[:len(malware_paths)]
print(len(malware_paths), len(benign_paths))

paths = malware_paths + benign_paths
labels = [1]*len(malware_paths) + [0]*len(benign_paths)
train_paths, test_paths, _, _ = train_test_split(
    paths,
    labels,
    test_size=0.3,
    random_state=SEED,
    stratify=labels
)

model = DGCNN(
    train_paths=train_paths,
    test_paths=test_paths,
    report_path=f'reports/RQ1/{args.input}.txt'
)
model.run()
