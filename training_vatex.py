import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
from keras.preprocessing.sequence import pad_sequences
from os import path
import json
from glob import glob

from video_caption import build_model
from data_generator import data_generator_vatex
from data_generator import preprocessing_captions_vatex

def main():

	file = open('vatex_training_v1.0.json',)
	captions = json.load(file)
	file.close()

	features_files = glob('vatex_features/*')

	captions = preprocessing_captions_vatex(captions)

	tokenizer = keras.preprocessing.text.Tokenizer(filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n')
	maxlen = float('-inf')

	cap_str = []
	for i in captions.values():
		cap_str+=i

	tokenizer.fit_on_texts(cap_str)
	cap_tokens = tokenizer.texts_to_sequences(cap_str)

	for i in cap_tokens:
		maxlen = max(maxlen,len(i))

	del cap_str
	del cap_tokens


	gen = data_generator_vatex(features_files,captions,tokenizer,maxlen)
	vocab_dim = len(tokenizer.word_index)+1

	model = build_model(1024,maxlen,vocab_dim)


	model.fit(gen)

if __name__ == '__main__':
	main()






