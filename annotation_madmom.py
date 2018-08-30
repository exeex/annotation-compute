import numpy as np
import os
import matplotlib.pyplot as plt
import madmom
import sys
from concurrent.futures import ProcessPoolExecutor

if __name__ == '__main__':
    # Input: Folder Path
    music_folder = "song_file"
    npy_folder = "annotation_file_madmom"
    if not os.path.exists(music_folder):
        os.mkdir(music_folder)
    if not os.path.exists(npy_folder):
        os.mkdir(npy_folder)
    mp3files = os.listdir(music_folder)
    for mfile in mp3files:
        song_id = mfile.split('-')[0]
        outfile_folder = os.path.join(npy_folder, song_id)
        if not os.path.exists(outfile_folder):
            os.mkdir(outfile_folder)


def process(max_workers=4):
    infile_paths = [os.path.join(music_folder, file) for file in mp3files]
    outfile_paths = [os.path.join(os.path.join(npy_folder, file.split('-')[0]), file.split('-')[0]+'-madmom_onset.npy') for file in mp3files]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for idx, data in enumerate(executor.map(_process, infile_paths)):
            outfile = outfile_paths[idx]
            np.save(outfile, data)


def _process(file):
    print(file, file=sys.stderr)
    try:
        # madmom_onset
        proc = madmom.features.onsets.CNNOnsetProcessor()(file)
        # print(proc)
        # plt.plot(proc[:1000])
        # plt.show()
        act = madmom.features.onsets.OnsetPeakPickingProcessor()
        # print(act(proc))
        return act(proc)
    except Exception as err:
        print("load file %s fail," % file, err)
        return None


def load_npy():
    npy_subfolders = os.listdir(npy_folder)
    for npy_subfolder in npy_subfolders:
        npy_subfolder_path = os.path.join(npy_folder, npy_subfolder)
        npy_files = os.listdir(npy_subfolder_path)
        outfile_paths = [os.path.join(npy_subfolder_path, file) for file in npy_files]
        # print(outfile_paths)
        for path in outfile_paths:
            outfile_name = path.split('\\')[-1]
            fnpy = np.load(path)
            print(outfile_name, fnpy)


if __name__ == '__main__':
    proc = process(max_workers=3)
    # ld = load_npy()
