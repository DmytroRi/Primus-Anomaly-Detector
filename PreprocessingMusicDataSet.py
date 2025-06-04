import math
import os
import librosa
import json
import numpy as np
import datetime
import ConnectionDB as DB


DATASET_PATH = "E:\\dataset"
JSON_PATH    = "computed_data/features_13mfcc_20frame_10hop_delta_deltadelta_nocmvn.json"
USE_DB       = True
EXCEPTIONS_PATH = "data_set/Exceptions.txt"  # Path to the exceptions file

# Constants
SAMPLE_RATE  = 15000 
N_MFCC       = 20
FRAME_SMP    = 4096
HOP_SMP      = 1024
SILENCE_THRESHOLD_DB = 40
DUMMY_CLASSIFICATION = 8

def read_exceptions():
    """Reads exceptions files"""
    print("Reading exceptions from file...")
    with open(EXCEPTIONS_PATH, 'r') as file:
        lines = file.readlines()
    lines = [line.strip() for line in lines]
    return lines

def print_genre_info(genre, length_seconds, length_frames, num_files):
    """Prints information about the genre."""
    print(f"Genre: {genre}")
    print(f"  Length (h:mm:ss):\t {str(datetime.timedelta(seconds=float(f'{length_seconds:.2f}')))}")
    print(f"  Length (frames):\t {length_frames}")
    print(f"  Number of files:\t {num_files}")
    print("-" * 40)
    
def trim_silence_edges(signal):

    intervals = librosa.effects.split(y=signal, top_db=SILENCE_THRESHOLD_DB)

    if intervals.size != 0:
        start_sample = intervals[0, 0]  # start of the first segment
        end_sample   = intervals[-1, 1] # end of the last segment
        signal_trimmed = signal[start_sample:end_sample]
        #print(f"Trimmed {start_sample/SAMPLE_RATE:.2f}s of silence/intros and {len(signal)-end_sample/SAMPLE_RATE:.2f}s of silence/outros.")
    else:
        signal_trimmed = signal

    return signal_trimmed

def compute_mean_variance(features):
    """Compute mean and variance of the features."""
    mean = np.mean(features, axis=1)        # shape: (n_feat_total,)
    variance = np.var(features, axis=1)     # shape: (n_feat_total,)
    agg = np.hstack((mean, variance))       # shape: (n_feat_total*2,)
    return agg

def extract_features(
        audio, sr= 5000,
        n_mfcc=20, n_mels=40, 
        frame_smp=20, hop_smp=10,
        pre_emphasis=0.97, lifter=22,
        with_delta=True, with_delta_delta=True,
        cmvn=True):
    """Extract MFCC features from a raw audio signal, using sample based framing."""
    # 1) pre-emphasis
    audio = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])

    # 2) mel spectrogram
    S = librosa.feature.melspectrogram(
        y=audio, sr=sr,
        n_fft=frame_smp, hop_length=hop_smp,
        n_mels=n_mels, window='hann'
    )

    # 3) log + DCT → MFCC
    mfcc = librosa.feature.mfcc(
        S=librosa.power_to_db(S),
        n_mfcc=n_mfcc
    )

    # 4) spectral descriptors
    cent = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=frame_smp, hop_length=hop_smp)

    # 5) spectral bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, n_fft=frame_smp, hop_length=hop_smp)

    # 6) spectal roll-off
    roll = librosa.feature.spectral_rolloff(y=audio, sr=sr, n_fft=frame_smp, hop_length=hop_smp, roll_percent=0.85)

    # 7) root mean square energy
    rmse = librosa.feature.rms(y=audio, frame_length=frame_smp, hop_length=hop_smp, center=True)

    # 8) zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y=audio, frame_length=frame_smp, hop_length=hop_smp, center=True)

    # 9) dynamic tempo
    oenv = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=hop_smp, aggregate= np.median)
    tempo = librosa.beat.tempo(onset_envelope=oenv, sr=sr, hop_length=hop_smp, aggregate=None).reshape(1, -1) # shape: (1, n_frames)

    features = np.vstack([mfcc, cent, bandwidth, roll, rmse, zcr, tempo]) # features.shape = (n_feat_total, n_frames)

    # 10) mean and variance
    #features = compute_mean_variance(features) # features.shape = (n_feat_total*2, n_frames)

    return features  

def save_features(dataset_path, json_path):
    print("Execution of save_mfcc function has started.")

    exceptions = read_exceptions()
    data = {}

    for dirpath, _, filenames in os.walk(dataset_path):
        if dirpath == dataset_path:
            continue
        
        genre = os.path.basename(dirpath)
        if genre == "nu_metal" or genre == "alternative_metal":
            continue
        data.setdefault(genre, {})
        print(f"Processing {genre}\n")

        lenght_seconds = 0
        length_frames = 0

        filtered_filenames = [fname for fname in filenames if fname not in exceptions]
        # Process all files (songs) in the folder
        for fname in filtered_filenames:
            file_path = os.path.join(dirpath, fname)
            signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
            
            #print(f"Processing {fname}...")
            # Trim silence from the beginning of the audio signal
            signal = trim_silence_edges(signal)

            lenght_seconds += librosa.get_duration(y=signal, sr=sr)
            length_frames += math.ceil(len(signal) / HOP_SMP)

            # Check if the signal is empty after trimming
            if signal.size == 0:
                print(f"File {file_path} is empty, skipping.")
                continue

            # Compute MFCC: shape = (n_mfcc, n_frames)
            feature_frames = extract_features(audio=signal,
                                       sr=SAMPLE_RATE,
                                       n_mfcc=N_MFCC,
                                       frame_smp=FRAME_SMP,
                                       hop_smp=HOP_SMP,
                                       pre_emphasis=0.97,
                                       lifter=22,
                                       with_delta=False,
                                       with_delta_delta=False,
                                       cmvn=False)
            
            #feature_frames = feature_frames.T  # Transpose to shape (n_frames, n_mfcc)
            
            records = []
            for i in range(feature_frames.shape[0]):
                feature_values = feature_frames[i].tolist()
                records.append((
                fname,                   # song_name
                genre,                   # song_genre
                DUMMY_CLASSIFICATION,    # classification
                feature_values           # list of features
                ))
            DB.insert_features_V2(records)
            print(f"Inserted {len(records)} frames for {fname}.")    
            print(f"Processed {fname} with {feature_frames.shape[0]} frames and {feature_frames.shape[1]} Features.")

        print_genre_info(genre, lenght_seconds, lenght_seconds, len(filenames))


if __name__ == "__main__":
    save_features(DATASET_PATH, JSON_PATH)
