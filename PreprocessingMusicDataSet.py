import math
import os
import librosa
import json

DATASET_PATH = "music_data/src/"
JSON_PATH = "music_data/primus_data.json"
SAMPLE_RATE = 25050

def save_mfcc(dataset_path, json_path, n_mfcc=40, n_fft=2048, hop_length=512, segment_duration=15):
    print("Execution of save_mfcc function has started.\n")
    data = {
        "mapping": [],
        "mfcc": [],
        "labels": []
    }
    # Traverse through all subdirectories (each representing a genre)
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        # Skip the root folder; process only subdirectories
        if dirpath != dataset_path:
            # Extract the genre (semantic label) from the directory name
            semantic_label = os.path.basename(dirpath)
            data["mapping"].append(semantic_label)
            print(f"Processing {semantic_label}\n")
            
            # Process all files in the genre folder
            for f in filenames:
                file_path = os.path.join(dirpath, f)
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                
                # Skip files that are too short for processing
                if len(signal) < n_fft:
                    print(f"File {file_path} is too short, skipping.")
                    continue

                track_duration = librosa.get_duration(y=signal, sr=sr)
                num_segments = math.floor(track_duration / segment_duration)
                if num_segments == 0:
                    num_segments = 1  # Ensure at least one segment
                
                # Calculate the number of samples per segment
                num_samples_per_segment = int(len(signal) / num_segments)
                expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)

                # Process segments extracting MFCCs
                for s in range(num_segments):
                    start_sample = s * num_samples_per_segment
                    finish_sample = start_sample + num_samples_per_segment
                    segment = signal[start_sample:finish_sample]
                    
                    if len(segment) < n_fft:
                        print(f"Segment {s} in file {file_path} is too short for n_fft, skipping.")
                        continue
                    
                    mfcc = librosa.feature.mfcc(y=segment,
                                                sr=sr,
                                                n_fft=n_fft,
                                                n_mfcc=n_mfcc,
                                                hop_length=hop_length)
                    
                    # Check if MFCC extraction resulted in the expected number of vectors (time frames)
                    if mfcc.shape[1] == expected_num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        # Use (i - 1) since the first directory (i==0) was the root; adjust if needed.
                        data["labels"].append(f)
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
    print("The end of the execution.\n")

if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, segment_duration=15)
