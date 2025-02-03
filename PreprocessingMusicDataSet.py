import math
import os
import librosa
import json

DATASET_PATH = "music_data/src/primus"
JSON_PATH = "music_data/primus_data.json"
SAMPLE_RATE = 25050
#DURATION = 30
#SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

def save_mfcc(dataset_path, json_path, n_mfcc=40, n_fft=2048, hop_length=512, segments_duration = 10):
    print("Execution of save_mfcc funtcion has started.\n")
    # dictionary to store data
    data = {
        "mapping":  [],
        "mfcc":     [],
        "labels":   []
    }
    #look through all the genres 
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        if dirpath == dataset_path:
            # save the semantic label
            dirpath_components = dirpath.split("/")
            semantic_label = dirpath_components[-1]
            data["mapping"].append(semantic_label)

            #process files for a specific genre
            print("Processing {}\n". format(semantic_label))
            for f in filenames:
                # load audio file
                file_path = os.path.join(dirpath, f)
                signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)
                
                 # Skip files that are too short overall
                if len(signal) < n_fft:
                    print(f"File {file_path} is too short, skipping.")
                    continue

                # calculation of samples number per segment
                track_duration = librosa.get_duration(y = signal)
                
                # calculate number of segments from track duration and desired segment duration.
                num_segments = int(track_duration / segments_duration)
                if num_segments == 0:
                    num_segments = 1  # Ensure at least one segment               
                
                 # calculate the number of samples per segment based on the full signal length.
                num_samples_per_segment = int(len(signal) / num_segments)
                expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)

                # propcess sefmants extracting mfcc and storing data
                for s in range(num_segments):
                    start_sample = num_samples_per_segment * s
                    finish_sample = start_sample + num_samples_per_segment
                    
                    # Ensure the segment does not exceed the signal length
                    segment = signal[start_sample:finish_sample]
                    
                    # Check that the segment has enough samples for the FFT window
                    if len(segment) < n_fft:
                        print(f"Segment {s} in file {file_path} is too short for n_fft, skipping.")
                        continue
                    
                    mfcc = librosa.feature.mfcc(y = signal[start_sample : finish_sample],
                                                sr = sample_rate,
                                                n_fft = n_fft,
                                                n_mfcc = n_mfcc,
                                                hop_length = hop_length)
                    
                    #store mfcc for segment if has the expected length
                    if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i-1)
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
    print("The end of the execution.\n")

if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, segments_duration=10)