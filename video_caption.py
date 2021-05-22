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


def greedy_search(input_vid,seq,model,tokenizer,maxlen):

		output = model.predict([np.expand_dims(input_vid,axis=0),seq])
		output = output.flatten()
		#print(seq)
		arg_output = np.argsort(output)

		best_token = arg_output[-1]

		return best_token


class BahdanauAttention(tf.keras.Model):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, features, hidden):
    # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

    # hidden shape == (batch_size, hidden_size)
    # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
    hidden_with_time_axis = tf.expand_dims(hidden, 1)

    # attention_hidden_layer shape == (batch_size, 64, units)
    attention_hidden_layer = (tf.nn.tanh(self.W1(features) +
                                         self.W2(hidden_with_time_axis)))

    # score shape == (batch_size, 64, 1)
    # This gives you an unnormalized score for each image feature.
    score = self.V(attention_hidden_layer)

    # attention_weights shape == (batch_size, 64, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * features
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights


def build_seq(input_vid,model,tokenizer,maxlen):
	seq = tokenizer.texts_to_sequences(['<beg>'])[0]
	seq = np.asarray(seq)
	seq = np.expand_dims(seq,axis=0)
	seq = pad_sequences(seq,maxlen=maxlen,padding='post')

	for i in range(1,maxlen):
		#print(seq)
		token = greedy_search(input_vid,seq,model,tokenizer,maxlen)
		seq[0,i] = token

		if token == tokenizer.word_index['<end>']:
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


def build_model_add_outputs(input_len,maxlen,vocab_dim):

	encoder_input = layers.Input(shape=(None,input_len),name='enc_inp')
	encoder_output,state_c,state_h = layers.LSTM(128,return_state=True)(encoder_input)

	states = [state_c,state_h]


	decoder_partial_caption = layers.Input(shape=(maxlen,),name='partial_caption')
	embedding = layers.Embedding(input_dim=vocab_dim,output_dim=64)(decoder_partial_caption)

	rnn_output = layers.LSTM(128,name='rnn_op')(embedding,initial_state=states)

	decoder_output = layers.add([rnn_output,encoder_output])


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


def main():

	model = build_model(1024,35,8)


if __name__ == '__main__':
	main()








