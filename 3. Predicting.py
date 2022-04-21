import tensorflow as tf
import librosa

SONG_PATH = "test_country.wav"
SONG_NAME = "1"

NUM_MFCC = 13
N_FFT = 2048
HOP_LENGTH= 512
SAMPLE_RATE = 22050
SAMPLES_PER_TRACK = SAMPLE_RATE * 30
NUM_SEGMENT = 10

CLASSES = ["Blues", "Classical", "Country", "Disco", "Hiphop",
               "Jazz", "Metal", "Pop", "Reggae", "Rock"]

SAMPLES_PER_SEGMENT = int(SAMPLES_PER_TRACK / NUM_SEGMENT)

def check_duration(song_length):
    global flag
    global parts
    flag = 0
    if song_length > 30:
        print("Song is greater than 30 seconds")
        samples_per_track_30 = SAMPLE_RATE * song_length
        parts = int(song_length/30)
        global samples_per_segment_30
        samples_per_segment_30= int(samples_per_track_30 / (parts))
        flag = 1
        print("Song sliced into "+str(parts)+" parts\n")

    elif song_length == 30:
        parts = 1
        flag = 0
    else:
        print("Too short, enter a song of length minimum 30 seconds")
        flag = 2


def create_mfccs():
    class_predictions = []

    for i in range(0, parts):
        print(i)
        if flag == 1:
            print("Song snippet ", i + 1)
            start30 = samples_per_segment_30 * i
            finish30 = start30 + samples_per_segment_30
            y = x[start30:finish30]
        elif flag == 0:
            print("Song is 30 seconds, no slicing")

        for n in range(NUM_SEGMENT):
            start = SAMPLES_PER_SEGMENT * n
            finish = start + SAMPLES_PER_SEGMENT
            mfcc = librosa.feature.mfcc(y=y[start:finish], sr=SAMPLE_RATE, n_mfcc=NUM_MFCC, n_fft=N_FFT,
                                        hop_length=HOP_LENGTH)
            mfcc = mfcc.T
            mfcc = mfcc.reshape(1, mfcc.shape[0], mfcc.shape[1])
            array = model.predict(mfcc) * 100
            array = array.tolist()

            # find maximum percentage class predicted
            class_predictions.append(array[0].index(max(array[0])))

    return class_predictions

def get_prediction(class_predictions):
    prediction_per_part = []
    occurence_dict = {}
    for i in class_predictions:
        if i not in occurence_dict:
            occurence_dict[i] = 1
        else:
            occurence_dict[i] +=1
    max_key = max(occurence_dict, key=occurence_dict.get)
    prediction_per_part.append(CLASSES[max_key])
    prediction = max(set(prediction_per_part), key = prediction_per_part.count)
    print(f"\nGenre : {prediction}\n")

if __name__ == "__main__":

    # Loading the RNN-LTSM Model
    model = tf.keras.models.load_model("Genre_Classification_RNN_LSTM_Model.h5")

    # Loading Test Song
    x, sr = librosa.load(SONG_PATH, sr=SAMPLE_RATE)

    # Checking Duration of the Song
    song_length = int(librosa.get_duration(filename=SONG_PATH))
    check_duration(song_length)

    # Generating MFCCs
    class_predictions = create_mfccs()

    # Predicting
    get_prediction(class_predictions)

