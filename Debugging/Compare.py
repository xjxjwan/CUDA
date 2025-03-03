# compare two .txt files to check whether they are exactly the same

import hashlib


def get_file_hash(file_path):

    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def compare_files(file1, file2):

    hash1 = get_file_hash(file1)
    hash2 = get_file_hash(file2)

    if hash1 == hash2:
        print("Same")
    else:
        print("Different")


if __name__ == "__main__":
    
    name = "fluxX"
    file1 = "CPU_Case1_" + name + ".txt"
    file2 = "GPU_Case1_" + name + ".txt"
    compare_files(file1, file2)
