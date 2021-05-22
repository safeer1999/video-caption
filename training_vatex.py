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
from video_caption import build_model_add_outputs
from video_caption import build_seq
from video_caption import beam_search
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
	model.save('model.h5')


def main1():

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


	vocab_dim = len(tokenizer.word_index)+1

	model = keras.models.load_model('model.h5')

	for i in range(100):
	
		sample = np.load(features_files[i])
		sample = np.squeeze(sample,axis=0)

		text = build_seq(sample,model,tokenizer,maxlen)

		print(text)


def main2():

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


	vocab_dim = len(tokenizer.word_index)+1

	model = keras.models.load_model('model.h5')

	input_vid = np.load(features_files[100])
	input_vid = np.squeeze(input_vid,axis=0)
	test_cap = captions[features_files[100][15:-4]][2]
	test_cap = tokenizer.texts_to_sequences([test_cap])[0]



	for i in range(2,len(test_cap)+1):
		seq = pad_sequences([test_cap[:i]],maxlen,padding='post')
		output = model.predict([np.expand_dims(input_vid,axis=0),seq])
		output = output.flatten()
		token_id = np.argmax(output)
		token = tokenizer.sequences_to_texts([[token_id]])[0]
		print(token)




if __name__ == '__main__':
	main()






