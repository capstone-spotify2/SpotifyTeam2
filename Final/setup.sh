#!/bin/bash
echo $STRING
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/pipeline/1-LSTM_model_for_tag_prediction/."
echo $DIR
pip install django==1.8.13
pip install gensim
pip install tensorflow
pip install keras