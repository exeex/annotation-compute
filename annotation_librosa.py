import numpy as np
import librosa
import os
from concurrent.futures import ProcessPoolExecutor
import sys

if __name__ == '__main__':
    # Input: Folder Path
    music_folder = "song_file"
    npy_folder = "annotation_file_librosa"
    if not os.path.exists(music_folder):
        os.mkdir(music_folder)
    if not os.path.exists(npy_folder):
        os.mkdir(npy_folder)
    # os.makedirs(music_folder, exist_ok=True)
    # os.makedirs(npy_folder, exist_ok=True)

    mp3files = os.listdir(music_folder)


def process(fortest=False, max_workers=4):
    infile_paths = [os.path.join(music_folder, file) for file in mp3files]
    outfile_paths_sr = []
    outfile_paths_duration = []
    outfile_paths_tempo = []
    outfile_paths_beat_times = []
    outfile_paths_mfcc = []
    outfile_paths_chromagram = []
    outfile_paths_onset = []

    for file in mp3files:
        song_id = file.split('-')[0]
        outfile_folder = os.path.join(npy_folder, song_id)
        if not os.path.exists(outfile_folder):
            os.mkdir(outfile_folder)

        path_sr = os.path.join(outfile_folder, song_id + '-librosa_sr.npy')
        outfile_paths_sr.append(path_sr)
        path_duration = os.path.join(outfile_folder, song_id + '-librosa_duration.npy')
        outfile_paths_duration.append(path_duration)
        path_tempo = os.path.join(outfile_folder, song_id + '-librosa_tempo.npy')
        outfile_paths_tempo.append(path_tempo)
        path_beat_times = os.path.join(outfile_folder, song_id + '-librosa_beat_times.npy')
        outfile_paths_beat_times.append(path_beat_times)
        path_mfcc = os.path.join(outfile_folder, song_id + '-librosa_mfcc.npy')
        outfile_paths_mfcc.append(path_mfcc)
        path_chromagram = os.path.join(outfile_folder, song_id + '-librosa_chromagram.npy')
        outfile_paths_chromagram.append(path_chromagram)
        path_onset = os.path.join(outfile_folder, song_id + '-librosa_onset.npy')
        outfile_paths_onset.append(path_onset)

    if fortest:
        data = _process(librosa.util.example_audio_file())
        return data

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for idx, data in enumerate(executor.map(_process, infile_paths)):
            file_sr = outfile_paths_sr[idx]
            np.save(file_sr, data[0])
            file_duration = outfile_paths_duration[idx]
            np.save(file_duration, data[1])
            file_tempo = outfile_paths_tempo[idx]
            np.save(file_tempo, data[2])
            file_beat_times = outfile_paths_beat_times[idx]
            np.save(file_beat_times, data[3])
            file_mfcc = outfile_paths_mfcc[idx]
            np.save(file_mfcc, data[4])
            file_chromagram = outfile_paths_chromagram[idx]
            np.save(file_chromagram, data[5])
            file_onset = outfile_paths_onset[idx]
            np.save(file_onset, data[6])


def _process(file):
    print(file, file=sys.stderr)
    try:
        y, sr = librosa.load(file, sr=None)

        # Get duration
        duration = librosa.get_duration(y, sr)

        # Set the hop length
        hop_length = 512

        # Separate harmonics and percussives into two waveforms
        y_harmonic, y_percussive = librosa.effects.hpss(y)

        # Beat track on the percussive signal
        tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=sr)

        # Convert the frame indices of beat events into timestamps
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)

        # Compute MFCC features from the raw signal
        mfcc = librosa.feature.mfcc(y, sr, hop_length=hop_length, n_mfcc=13)

        # Compute chroma features from the harmonic signal
        chromagram = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)

        # segments_time
        onset_time = librosa.onset.onset_detect(y=y, sr=sr, units='time')
        return sr, duration, tempo, beat_times, mfcc, chromagram, onset_time

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
