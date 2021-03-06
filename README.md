Lyric-Based Music Tagging and Recommendation System
===================================================

Spotify provides millions of song tracks covering a full spectrum of music variety to users all over the world on multiple platforms. The key mission of Spotify is to help people find the right music at every moment through tailored song recommendations and playlists. We carry on this important spirit in our project and seek to further enhance users’ music enjoying experience. Specifically, as many research focus on acoustic features in music analysis, we aim to analyze the language component of songs, i.e. lyrics, to associate songs with relatable concepts such as moods and themes. We explore with supervised and unsupervised methods including Latent Dirichlet Allocation(LDA), Word2Vec and Long Short-Term Memory(LSTM) for language processing and predictive modeling. Our final model can be used to provide a two-way translation between songs and tags. In one direction, we have an automatic tagging system which suggests potential tags for any song. And conversely, we can recommend a list of songs associated with the keywords from user inputs. The lyrics based model yields satisfactory predictions and thus offers a new angle in analyzing song content. By leveraging the rich emotional and descriptive content encrypted in song lyrics, we provide Spotify a solution to create even more personalized music enjoying experience for its users.



> **Note: Please run all the functions under python 2**

Part 1: environment setup
-------------------------


1. Install all the packages by running the following shell script:

    ``./setup.sh``

2. Download pre-trained word2vec model from [GoogleNews-vectors-negative300.bin.gz](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing). 
Unzip the word2vec model and place the file inside this folder: `./Final_Deliverable/pipeline/1-LSTM_model_for_tag_prediction`.

> Note: if you have already installed django, gensim, tensorflow, or keras package, please make sure they have the following versions: <br />
django: 1.8.13,  <br />
gensim: 2.0.0,  <br />
tensorflow: 1.1.0,  <br />
keras: 2.0.4
    
Part 2: Run Tag Prediction Baseline Model in terminal
--------------------------------------------------
`cd ./Final_Deliverable/pipeline/0-baseline_model_for_tag_prediction/` to run the baseline model


### step 1: feature_engineering

Run feature_engineering.py by:
``python feature_engineering.py raw_data.csv [-f output.csv -n n_features]``

Example:
``python feature_engineering.py full_dataset.csv -f test.csv -n 150``

Parameters explanation: 

- `raw_data.csv` (input, required) - the file contains the information about all the songs in database. Required informations are (in order): Current_index, original_index, song_id, song_name, artist, tag, path_to_the_song lyrics_file, song_lyrics.

- `output.csv` (output, optimal) - the file returns the hashing vectorizer representation of the lyrics and tags needed for the modeling step. *default: "features.csv"*

- `n_features` (int, input, optimal) - number of dimensions of the hashing vectorizer that we want for the modeling step. *default: 150*


### step 2: train model:
Run model_training.py by:
``python model_training.py input_feature.csv output_model_name [-e est -d depth]``

Example:
``python model_training.py features.csv output_model -e 20 -d 5``

Parameters explanation:

- `input_feature.csv` (input, required) - the hashing vectorizer of the raw data generated by feature_engineering.py.

- `output_model_name` (output, required) - the name of output model we want to save for the fitting part.

- `est` (int, input, optimal) - number of estimators required for Random Forest model. *default: 15*

- `depth` (int, input, optimal) - max depth of the Random Forest model. *default: 5*



### step 3:  tag prediction:
Run model_fitting.py by:
``python model_fitting.py lyrics.txt -n n_features -m model``

Example:
``python model_fitting.py ../lyrics/If\ I\ die\ Young.txt -n 150 -m model.pickle``

Parameters explanation:
- `lyrics.txt` (input, required) - the lyrics txt file of the song we want to predict.

- `n_features` (input, optional) - number of dimensions of the hashing vectorizer, **must equivalent to n_features in training step**. *default: 150*

- `model` (input, optional) - the model we trained in model training step. *default: model.pickle - pre-trained model by the developer.*

Return value: 
- Terminal prints out 3 tags that have the top 3 probability.
- Example output: <br />
>The prediction results are:
>love
>memory
>sad


Part 3: Run Tag Prediction Advanced Model in terminal
------------------------------------------------------
`cd ./Final_Deliverable/pipeline/1-LSTM_model_for_tag_prediction/` to run the advanced model.
>**Note: Since the LSTM model is huge. It takes very long time to train and fit the model on a personal computer.**

### step 1: train model:
Run model_training.py by:
``python model_training.py output_model_name``

Example:
``python model_training.py output_model``

Parameters explanation:
- `output_model_name` (output, required) - the name of the output model.


### step 2: tag prediction:
Run model_fitting.py by:
``python model_fitting.py lyrics.txt model_name.h5``

Example:
``python model_fitting.py ../lyrics/If\ I\ die\ Young.txt output_model.h5``

Parameters explanation:
- `lyrics.txt` (input, required) - the lyrics txt file of the song we want to predict.

- `output_model.h5` (input, required) - the trained model from model_training.py 

Return value: 
- Terminal prints out 3 tags that have the top 3 probability.
- Example output: 
>The three top tags are:
>tag: funny
>tag: love
>tag: sad


Part 4: Run Song Recommendation System in terminal
------------------------------------------------------
`cd ./Final_Deliverable/pipeline/2-Recommendation/` to run the Recommendation model

### step 1: playlist recommendation:
Run tag_to_song.py by:
`python tag_to_song.py tag_name`

Example:
`python tag_to_song.py happy`

Parameters explanation:
- `tag_name` (input, required) - the name of tag or keywords that you want to get song playlist from.

Return value: 
- the top 10 songs that most related to the input tag or keywords.

Part 5: Setup and run User-interactive Website
--------------------------------------------------
`cd ./Final_Deliverable/website/mysite/` to start the web server. 

#### step 1: start the web server
`python manager.py runserver`

You will see similar output as below from terminal:<br /><br />
Performing system checks...<br />
<br />
System check identified no issues (0 silenced).<br />
May 14, 2017 - 08:54:41<br />
Django version 1.8.13, using settings 'mysite.settings'<br />
Starting development server at **http://127.0.0.1:8000/**<br />
Quit the server with CONTROL-C.
<br />
<br />
>type in chrome the following address:<br />
>http://127.0.0.1:8000/index/

You will see our webpage after loading for a while(3-5 min).

#### step 2: Tag Prediction Function:
type the artist and song name in the corresponding boxes and hit predict.<br />
<br />
The web will return the top 3 tags that associated with this song and the song lyrics
#### step 3: Song Recommendation:
type in the keywords that you want the song playlist relates to.<br />
<br />
The web will return list of 10 songs with the link to spotify song track.