import math
import os
import librosa
import json
import numpy as np

DATASET_PATH = "music_data/src/"
JSON_PATH = "computed_data/primus_data_mean.json"
SAMPLE_RATE = 25050
SILENCE_DETECTOR = -1131.37097      # Value of the first MFCC if the segment is silent

def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, segment_duration=15, take_mean=False):
    print("Execution of save_mfcc function has started.\n")
    # data will be a dictionary with genres as keys.
    data = {}

    # Variables to store information about the process
    total_duration = 0
    num_of_segments = 0

    # Traverse through all subdirectories (each representing a genre)
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        # Skip the root folder; process only subdirectories
        if dirpath != dataset_path:
            # Extract the genre from the directory name
            genre = os.path.basename(dirpath)
            print(f"Processing {genre}\n")
            
            # Initialize the genre entry if not already present
            if genre not in data:
                data[genre] = {}
            
            # Process all files (songs) in the folder
            for f in filenames:
                file_path = os.path.join(dirpath, f)
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                
                # Skip files that are too short for processing
                if len(signal) < n_fft:
                    print(f"File {file_path} is too short, skipping.")
                    continue

                segment_samples = int(sr * segment_duration)
                segments = [signal[i:i + segment_samples] for i in range(0, len(signal), segment_samples)]
                
                # Create a dictionary to hold segments for this song
                song_data = {}
                
                # Update state
                total_duration += librosa.get_duration(y=signal, sr=sr)
                num_of_segments += len(segments)

                # Process segments extracting MFCCs
                for seg_idx, segment in enumerate(segments):
                    # Process only segments that are exactly segment_duration long
                    if len(segment) != segment_samples:
                        print(f"Segment {seg_idx} in file {file_path} is not exactly {segment_duration} seconds, skipping.")
                        num_of_segments -= 1
                        continue
                    
                    mfcc = librosa.feature.mfcc(y=segment,
                                                sr=sr,
                                                n_mfcc=n_mfcc,
                                                n_fft=n_fft,
                                                hop_length=hop_length)
                    
                    if(take_mean):
                        # Aggregate MFCCs by taking the mean across time frames, yielding a 13-dimensional vector
                        mfcc = np.mean(mfcc, axis=1)
                    
                    # Slip silent segemts 
                    if mfcc[0] == SILENCE_DETECTOR:
                        print(f"Segment {seg_idx} in file {file_path} is silent (MFCC[0]={SILENCE_DETECTOR}), skipping.")
                        num_of_segments -= 1
                        continue

                    # Save MFCCs for this segment using the segment index as key
                    song_data[str(seg_idx)] = mfcc.tolist()
                
                # Under the current genre, use song data using the song filename as the key
                data[genre][f] = song_data

    # Write the restructured data into the JSON file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
    print("The end of the execution.\n")

    # Information about execution 
    print("-------------------------------------------")
    print(f"Total duration of tracks: {total_duration} seconds.\nTotal amount of segments: {num_of_segments}.\nSegments duration: {segment_duration} seconds.")
    print(f"Total effective duration: {segment_duration*num_of_segments} seconds.\nRetention rate: {100*segment_duration*num_of_segments/total_duration}%.")
    print("-------------------------------------------")

if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, segment_duration=15, take_mean=True)
