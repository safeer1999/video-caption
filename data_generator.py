import pickle
import numpy as np 
import keras
from keras.preprocessing.sequence import pad_sequences
import json
from glob import glob
# captions = ['the big bang theory',
# 	'sheldon is in the kitchen',
# 	'sheldon is making snow cones',
# 	'leonard walks into the hall',
# 	'leonard asks for a snow cone',
# 	'leonard thinks the snow cone is tasty',
# 	'leonard guesses the flavour of the snow cone',
# 	'sheldon tells leonard its mango caterpillar',
# 	'leonard spits the snow cone',
# 	'leonard walk away from the hall']


# features = pickle.load(open('test_dataset.pkl','rb'))

# tokenizer = keras.preprocessing.text.Tokenizer()
# tokenizer.fit_on_texts(captions)
#print(tokenizer.texts_to_sequences(captions))

#check for error nan loss mismatch in independent and dependent features

def preprocessing_captions_vatex(captions):
	
	processed = {}

	for i in range(len(captions)):
		cap = captions[i]['enCap']
		cap = list(map(lambda x: '<beg> ' + x + ' <end>',cap))

		processed[captions[i]['videoID']] = cap

	return processed

def data_generator(features,captions,tokenizer,maxlen):

	
	for i in range(len(features)-1):
		X = np.empty(shape=(0,7*7*2048))
		X_caption = np.empty(shape=(0,maxlen))
		y = np.asarray([])
		caption = tokenizer.texts_to_sequences([captions[i]])
		caption = np.asarray(caption)
		for j in range(1,caption.shape[1]+1):

			X = np.append(X,features[i],axis=0)
			seq = pad_sequences(caption[0,:j].reshape(1,-1),maxlen=maxlen,padding='post')
			X_caption = np.append(X_caption,seq,axis=0)
			#print(caption)
			y = np.append(y,caption[0,j-1])

		#X = np.expand_dims(X,axis=0)
		X = np.vsplit(X,X.shape[0]//features[0].shape[0])
		X = list(map(lambda x : np.expand_dims(x,axis=0),X))
		X = np.vstack(X)
		#print(X.shape,X_caption.shape,y.shape)
		yield ([X,X_caption],y)


def data_generator_vatex(features_files,captions, tokenizer,maxlen,epochs=1):
	for epoch in range(epochs):

		for i in range(len(features_files)):

			video_id = features_files[i][15:-4]

			X = np.empty(shape=(0,1024))
			X_caption = np.empty(shape=(0,maxlen))
			y= np.asarray([])
			try:
				caption = tokenizer.texts_to_sequences(captions[video_id])
			except KeyError:
				continue

			video_mat = np.load(features_files[i])
			video_mat = np.squeeze(video_mat,axis=0)

			for j in range(len(caption)):

				for k in range(1,len(caption[j])):

					X = np.append(X,video_mat,axis=0)
					seq = pad_sequences([caption[j][:k]],maxlen,padding='post')
					X_caption = np.append(X_caption,seq,axis=0)
					y = np.append(y,caption[j][k])

			X = np.vsplit(X,X.shape[0]//video_mat.shape[0])
			X = list(map(lambda x : np.expand_dims(x,axis=0),X))
			X = np.vstack(X)
			#print(X.shape,X_caption.shape,y.shape)
			yield ([X,X_caption],y)



# gen = data_generator(features,captions,tokenizer,8)

# for x,y in gen:
# 	print(y)
# 	print(tokenizer.sequences_to_texts([y]))

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


	gen = data_generator_vatex(features_files,captions,tokenizer,maxlen)

	data = next(gen)





if __name__ == '__main__':
	main()

	

