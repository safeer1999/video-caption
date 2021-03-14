import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
from keras.preprocessing.sequence import pad_sequences
from data_generator import data_generator
from os import path

def beam_search(count,input_vid,seq,model,tokenizer,maxlen,beam_width=3):
	if count == beam_width:
		return 1

	output = model.predict([np.expand_dims(input_vid,axis=0),seq])
	output = output.flatten()
	arg_output = np.argsort(output)

	best_tokens = arg_output[-3:]
	max_score = float('-inf')
	tk = None
	for i in range(best_tokens.shape[0]):
		score = output[best_tokens[i]]
		arg_seq = np.where(seq.flatten()!=0)[0]
		new_seq = np.expand_dims(np.append(seq.flatten()[arg_seq],best_tokens[i]),axis=0)
		new_seq = pad_sequences(new_seq,maxlen=maxlen,padding='post')
		score*=beam_search(count+1,input_vid,new_seq,model,tokenizer,maxlen)
		if score > max_score:
			max_score=score
			tk =  best_tokens[i]

	return tk


def build_seq(input_vid,model,tokenizer,maxlen):
	seq = tokenizer.texts_to_sequences(['<BEG>'])[0]
	seq = np.asarray(seq)
	seq = np.expand_dims(seq,axis=0)
	seq = pad_sequences(seq,maxlen=maxlen,padding='post')

	for i in range(1,maxlen+1):
		print(seq)
		token = beam_search(0,input_vid,seq,model,tokenizer,maxlen)
		seq[0,i] = token

		if token == tokenizer.word_index['end']:
			break



	return tokenizer.sequences_to_texts(seq)

def build_model(input_len,maxlen,vocab_dim):


	encoder_input = layers.Input(shape=(None,input_len),name='enc_inp')
	encoder_output,state_c = layers.SimpleRNN(128,return_state=True)(encoder_input)

	encoder_state = state_c

	decoder_partial_caption = layers.Input(shape=(maxlen,),name='partial_caption')
	embedding = layers.Embedding(input_dim=vocab_dim,output_dim=64)(decoder_partial_caption)

	decoder_output = layers.SimpleRNN(128,name='dec_op')(embedding,initial_state=encoder_state)
	#decoder_output = layers.SimpleRNN(128,name='dec_op')(embedding)

	output = layers.Dense(vocab_dim)(decoder_output)

	model = keras.Model([encoder_input,decoder_partial_caption],output)
	model.summary()
	model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

	return model
	

# captions = ['the big bang theory',
# 	'sheldon is in the kitchen',
# 	'sheldon is making snow cones',
# 	'leonard walks into the hall',
# 	'leonard asks for a snow cone',
# 	'leonard thinks the snow cone is tasty',
# 	'leonard guesses the flavour of the snow cone',
# 	'sheldon tells leonard its mango caterpillar',
# 	'leonard spits the snow cone']



# captions = list(map(lambda x: '<BEG> '+x+' <END>',captions))


# features = pickle.load(open('test_dataset.pkl','rb'))

# print(len(features),len(captions))
# print('\n')

# tokenizer = keras.preprocessing.text.Tokenizer()
# tokenizer.fit_on_texts(captions)

# #print(tokenizer.word_index)

# maxlen = 8
# vocab_dim = len(tokenizer.word_index)+1
# model = None
# if not path.exists('video_captioning.h5'):
# 	encoder_input = layers.Input(shape=(None,7*7*2048),name='enc_inp')
# 	encoder_output,state_c = layers.SimpleRNN(128,return_state=True)(encoder_input)

# 	encoder_state = state_c

# 	decoder_partial_caption = layers.Input(shape=(maxlen,),name='partial_caption')
# 	embedding = layers.Embedding(input_dim=vocab_dim,output_dim=64)(decoder_partial_caption)

# 	decoder_output = layers.SimpleRNN(128,name='dec_op')(embedding,initial_state=encoder_state)
# 	#decoder_output = layers.SimpleRNN(128,name='dec_op')(embedding)

# 	output = layers.Dense(vocab_dim)(decoder_output)

# 	model = keras.Model([encoder_input,decoder_partial_caption],output)
# 	model.summary()
# 	model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

# 	generator = data_generator(features,captions,tokenizer,maxlen)
# 	model.fit(generator,epochs=2)
# 	model.save('video_captioning.h5')

# else:
# 	model = keras.models.load_model('video_captioning.h5')








