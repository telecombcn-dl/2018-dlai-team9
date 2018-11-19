import os
import sys

def files_in_subdirs(dataset_path):
    for folder, subs, files in os.walk(dataset_path):
        for filename in files:
            print(os.path.join(folder, filename))

if __name__ == '__main__':
    files_in_subdirs('/projects/retrieval/imagenet/ILSVRC2015/Data/DET/val/')

