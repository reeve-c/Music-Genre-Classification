# Music-Genre-Classification
I implemented a RNN-LTSM Model for Music Genre Classification, which predicts the genre of the song provided as an input to the model.

## Data Preprocessing
The first step in building this model is to traverse all the Sub-Folders (i.e. all the genres) in the Genre Dataset and extract the features from each and every audio file using MFCCs and store them into a Comprehensive JSON File. The dataset consists of 10 genres which are Blues, Classical, Country, Disco, Hiphop, Jazz, Metal, Pop, Reggae and Rock. 

Link to Download Dataset: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification

## Building and Training the Model
The model is trained using tensorflow and keras over it. The Model consists of an input layer, 2 LSTM layers, a dense layer and an output layer. I defined 3 functions to build and train the model - 
1. load_data: To load the the JSON file.
2. prepare_datsets: To Prepare the datasets for the Model.
3. build_model: To build the RNN-LTSM Model.

## Predicting the Genre
Prediction is done by splitting the input song into segments of 30 seconds each and then predicting the genre on each segment. The most predicted genre out of all predictions is taken as the overall predicition. The song I used for prediction is 'Take Me Home, Country Roads' by John Denver. The song has a duration of 194 seconds and was divided in 6 segments and the Predicted Genre was Country, which is correct.
