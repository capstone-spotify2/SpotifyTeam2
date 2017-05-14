Lyric-Based Music Tagging and Recommendation System
===================================================

Spotify provides millions of song tracks covering a full spectrum of music variety to users all over the world on multiple platforms. The key mission of Spotify is to help people find the right music at every moment through tailored song recommendations and playlists. We carry on this important spirit in our project and seek to help further enhance users’ music enjoying experience. Specifically, as many research focus on acoustic features in music analysis, we aim to analyze the language component of songs, i.e. lyrics, to associate songs with relatable concepts such as moods and themes. We explore with supervised and unsupervised methods including Latent Dirichlet Allocation(LDA), Word2Vec and Long Short-Term Memory(LSTM) for language processing and for modeling predictions. Our final model serves as an automatic tagging system which can either return the suggested tags for a specific song, or produce a list of associated songs with user keyword inputs. The lyrics based model yields satisfactory predictions and thus offers a new angle in analyzing song contents. We can expect that by leveraging mostly the rich emotional and descriptive content encrypted in song lyrics, Spotify should be able to provide better personalized music experience for its users.



> Note: Please run all the functions under python 2

Part 1: environment setup
-------------------------


1. Install all the packages by running the following shell script:

    ``./setup.sh``

2. Download pre-trained word2vec model from [GoogleNews-vectors-negative300.bin.gz](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing). 
Unzip the word2vec model and locate the file inside `pipeline/1-LSTM_model_for_tag_prediction` folder

> Note: if you have already installed django, gensim, tensorflow, or keras package, please make sure they are in the following versions: django: 1.8.13,  gensim: 2.0.0,  tensorflow: 1.1.0,  keras: 2.0.4
    
Part 2: Run Tag Prediction Baseline Model in terminal
--------------------------------------------------
`cd ./Final_Deliverable/0-baseline_model_for_tag_prediction/` to run the baseline model


### step 1: feature_engineering

Run feature_engineering.py by:
>``python feature_engineering.py raw_data.csv [-f output.csv -n n_features]``

Example:
> ``python feature_engineering.py full_dataset.csv -f test.csv -n 150``

Parameters explanation: 

- `raw_data.csv` (input, required) - the file contains the information about all the songs in database. Required informations are (in order): Current_index, original_index, song_id, song_name, artist, tag, path_to_the_song lyrics_file, song_lyrics.

- `output.csv` (output, optimal) - the file returns the hashing vectorizer representation of the lyrics and tags needed for the modeling step. *default: "features.csv"*

- `n_features` (int, input, optimal) - number of dimensions of the hashing vectorizer that we want for the modeling step. *default: 150*


### step 2: train model:
Run model_training.py by:
> ``python model_training.py input_feature.csv output_model_name [-e est -d depth]``

Example:
> ``python model_training.py features.csv output_model -e 20 -d 5``

Parameters explanation:

- `input_feature.csv` (input, required) - the hashing vectorizer of the raw data generated by feature_engineering.py.

- `output_model_name` (output, required) - the name of output model we want to save for the fitting part.

- `est` (int, input, optimal) - number of estimators required for Random Forest model. *default: 15*

- `depth` (int, input, optimal) - max depth of the Random Forest model. *default: 5*



### step 3:  tag prediction:
Run model_fitting.py by:
> ``python model_fitting.py lyrics.txt -n n_features -m model``

Example:
> ``python model_fitting.py ../lyrics/If\ I\ die\ Young.txt -n 150 -m model.pickle``

Parameters explanation:
- `lyrics.txt` (input, required) - the lyrics txt file of the song we want to predict.

- `n_features` (input, optional) - number of dimensions of the hashing vectorizer, **must equivalent to n_features in training step**. *default: 150*

- `model` (input, optional) - the model we trained in model training step. *default: model.pickle - pre-trained model by the developer.*

Return value: 
- Terminal prints out 3 tags that have the top 3 probability.
- Example output: 

    >The prediction results are:
         
        >love
            
        >memory
            
        >sad




Part 3: Run Tag Prediction Advanced Model in terminal
------------------------------------------------------
`cd ./Final_Deliverable/1-LSTM_model_for_tag_prediction/` to run the advanced model.
>Note: **Since the LSTM model is huge. It takes very long time to train and fit on personal computer.**

### step 1: train model:
Run model_training.py by:
> ``python model_training.py output_model_name``

Example:
> ``python model_training.py output_model``

Parameters explanation:
- `output_model_name` (output, required) - the name of the output model.


### step 2: tag prediction:
Run model_fitting.py by:
> ``python model_fitting.py lyrics.txt``

Example:
> ``python model_fitting.py ../lyrics/If\ I\ die\ Young.txt``

Parameters explanation:
- `lyrics.txt` (input, required) - the lyrics txt file of the song we want to predict.

Return value: 
- Terminal prints out 3 tags that have the top 3 probability.
- Example output: 
    >The three top tags are:
    
    >tag: funny
    
    >tag: love
    
    >tag: sad


Part 4: Run Song Recommendation System in terminal
------------------------------------------------------
`cd ./Final_Deliverable/2-Recommendation/` to run the Recommendation model

### step 1: playlist recommendation:
Run tag_to_song.py by:
>`python tag_to_song.py tag_name`

Example:
>`python tag_to_song.py happy`

Parameters explanation:
- tag_name (input, required) - the name of tag or keywords that you want to get song playlist from.

Return value: 
- the top 10 songs that most related to the input tag or keywords.

Part 5: Setup and run User-interactive Website
--------------------------------------------------