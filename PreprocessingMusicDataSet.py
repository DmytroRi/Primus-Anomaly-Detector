import math
import os
import librosa
import json
import numpy as np
import ConnectionDB as DB


DATASET_PATH = "E:\\dataset"
JSON_PATH    = "computed_data/features_13mfcc_20frame_10hop_delta_deltadelta_nocmvn.json"
USE_DB       = True

# Constants
SAMPLE_RATE  = 22050 
N_MFCC       = 13
FRAME_MS     = 20
HOP_MS       = 10
SILENCE_THRESHOLD_DB = 40
DUMMY_CLASSIFICATION = 8

def trim_silence_edges(signal):

    intervals = librosa.effects.split(y=signal, top_db=SILENCE_THRESHOLD_DB)

    if intervals.size != 0:
        start_sample = intervals[0, 0]  # start of the first segment
        end_sample   = intervals[-1, 1] # end of the last segment
        signal_trimmed = signal[start_sample:end_sample]
        print(f"Trimmed {start_sample/SAMPLE_RATE:.2f}s of silence/intros and {len(signal)-end_sample/SAMPLE_RATE:.2f}s of silence/outros.")
    else:
        signal_trimmed = signal

    return signal_trimmed

def extract_mfcc(
        audio, sr= 5000,
        n_mfcc=13, n_mels=40, 
        frame_ms=20, hop_ms=10,
        pre_emphasis=0.97, lifter=22,
        with_delta=True, with_delta_delta=True,
        cmvn=True):
    
    # 1) pre-emphasis
    audio = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])

    # 2) frame & window via n_fft & hop_length
    n_fft      = int(sr * frame_ms/1000)
    hop_length = int(sr * hop_ms/1000)

    # 3) mel spectrogram
    S = librosa.feature.melspectrogram(
        y=audio, sr=sr,
        n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels, window='hann'
    )

    # 4) log + DCT → MFCC
    mfcc = librosa.feature.mfcc(
        S=librosa.power_to_db(S),
        n_mfcc=n_mfcc
    )

    # 5a) liftering
    n_frames = mfcc.shape[1]
    n = np.arange(n_mfcc)
    lift = 1 + (lifter / 2) * np.sin(np.pi * n / lifter)
    mfcc *= lift[:, np.newaxis]

    # 5b) deltas
    if with_delta:
        d1 = librosa.feature.delta(mfcc, order=1)
        d2 = librosa.feature.delta(mfcc, order=2)
        mfcc = np.vstack([mfcc, d1])
        if with_delta_delta:
            mfcc = np.vstack([mfcc, d2])

    # 5c) CMVN
    if cmvn:
        mfcc = (mfcc - mfcc.mean(axis=1, keepdims=True)) \
               / (mfcc.std(axis=1, keepdims=True) + 1e-8)

    return mfcc  

def save_mfcc(dataset_path, json_path):
    print("Execution of save_mfcc function has started.")

    data = {}

     # precompute sample counts
    frame_length = int(SAMPLE_RATE * FRAME_MS / 1000)
    hop_length   = int(SAMPLE_RATE * HOP_MS   / 1000)

    for dirpath, _, filenames in os.walk(dataset_path):
        if dirpath == dataset_path:
            continue
        
        genre = os.path.basename(dirpath)
        data.setdefault(genre, {})
        print(f"Processing {genre}\n")
         
         
        # Process all files (songs) in the folder
        for fname in filenames:
            file_path = os.path.join(dirpath, fname)
            signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
            
            print(f"Processing {fname}...")
            # Trim silence from the beginning of the audio signal
            signal = trim_silence_edges(signal)

            # Check if the signal is empty after trimming
            if signal.size == 0:
                print(f"File {file_path} is empty, skipping.")
                continue

            # Compute MFCC: shape = (n_mfcc, n_frames)
            mfcc_frames = extract_mfcc(audio=signal,
                                       sr=SAMPLE_RATE,
                                       n_mfcc=N_MFCC,
                                       frame_ms=FRAME_MS,
                                       hop_ms=HOP_MS,
                                       pre_emphasis=0.97,
                                       lifter=22,
                                       with_delta=False,
                                       with_delta_delta=False,
                                       cmvn=False)

            # Reshape MFCC: shape = (n_frames, n_mfcc)
            mfcc_frames = mfcc_frames.T
            
            if USE_DB:
                records = []
                for i in range(mfcc_frames.shape[0]):
                    mfcc_values = mfcc_frames[i].tolist()
                    records.append((
                    fname,                   # song_name
                    genre,                   # song_genre
                    DUMMY_CLASSIFICATION,    # classification
                    mfcc_values              # list of 13 floats
                ))
                DB.insert_in_DB(records)
                print(f"Inserted {len(records)} frames for {fname}.")    
            else:
                data[genre][fname] = {"frames": mfcc_frames.tolist()}
            print(f"Processed {fname} with {mfcc_frames.shape[0]} frames and {mfcc_frames.shape[1]} Features.")

    if not USE_DB:
        # Write the restructured data into the JSON file
        print(f"Writing data to {json_path}...\n")
        with open(json_path, "w") as fp:
            json.dump(data, fp, indent=2)
    print("The end of the execution.\n")

    # Information about execution 
    # print("-------------------------------------------")
    # print(f"Total duration of tracks: {total_duration} seconds.\nTotal amount of segments: {num_of_segments}.\nSegments duration: {segment_duration} seconds.")
    # print(f"Total effective duration: {segment_duration*num_of_segments} seconds.\nRetention rate: {100*segment_duration*num_of_segments/total_duration}%.")
    # print("-------------------------------------------")

if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH)
