import os
import hashlib
from glob import glob
from joblib import Parallel, delayed
from multiprocessing import cpu_count


N_JOBS = cpu_count()
BEG_ELF_PATHS = glob('../IoT_Benign_Collection_2021/*')


def md5(file_path):
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            md5_hash.update(byte_block)
    return md5_hash.hexdigest()


def rename(file_path):
    sha256_hash = file_path.split('/')[-1]
    md5_hash = md5(file_path)
    if os.path.exists(graph_path := f'data/benign/{md5_hash}.txt'):
        os.rename(graph_path, f'data/benign/{sha256_hash}.txt')
        print(graph_path)
        return 1
    return 0


output = Parallel(N_JOBS)(delayed(rename)(file_path)
                          for file_path in BEG_ELF_PATHS)
print(f'Renamed {sum(output)} files.')
