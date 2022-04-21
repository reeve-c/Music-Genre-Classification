import json
import os
import math
import librosa

def save_mfcc(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):

    # The number of samples of audio recorded every second
    SAMPLE_RATE = 22050

    # Duration of audio file (Measured in seconds)
    TRACK_DURATION = 30

    SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION
    
    # Dictionary to store Mapping, Labels, and MFCCs
    data = {
        "mapping": [],
        "labels": [],
        "mfcc": []
    }

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    # loop through all genre sub-folder
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # Ensure we're processing a genre sub-folder level
        if dirpath is not dataset_path:

            # Save genre label (i.e., sub-folder name) in the mapping
            semantic_label = dirpath.split("/")[-1]
            data["mapping"].append(semantic_label)
            print("\nProcessing: {}".format(semantic_label))

            # Processing all audio files in genre sub-dir
            for f in filenames:

                # loading Audio file
                file_path = os.path.join(dirpath, f)
                signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

                # Process all segments of audio file
                for d in range(num_segments):

                    # Calculating Start and Finish sample for current segment
                    start = samples_per_segment * d
                    finish = start + samples_per_segment

                    # Extracting MFCCs
                    mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
                    mfcc = mfcc.T

                    # Storing only mfcc feature with expected number of vectors
                    if len(mfcc) == num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i-1)
                        print("{}, segment:{}".format(file_path, d+1))

    # Savinf MFCCs to json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

if __name__ == "__main__":

    # Path to the Dataset
    DATASET_PATH = "Music_Genre_Dataset"

    # Path to json file used to save MFCCs
    JSON_PATH = "json_data_file.json"

    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)

