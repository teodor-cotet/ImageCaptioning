#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


# Small LSTM Network to Generate Text for Alice in Wonderland
import numpy as np
import keras
from keras.layers import Input
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint, LambdaCallback, Callback
from keras.utils import np_utils

#encoder stuff
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model

from gensim.models.wrappers import FastText as FastTextWrapper
import nltk
nltk.download('punkt')


# In[2]:


# hyper params
seq_length = 17
img_embedding = 300
nr_input_lines = 5000 # used to determine steps_per_epoch (nr batches)
photos_per_batch = 5 # a batch will consists of photos_per_batch photos (each photo has 20-30 text samples)
lstm_cell = 128
examples_train = 0


# In[3]:


filepath = "dataset_txt/Flickr8k.lemma.token.txt"
fast_text = "fastText_eng/wiki.simple"
dir_imgs = "dataset_img"
token_start = 'sstart'
token_end = 'eend'
special_token = 'xx'
# load fast text - takes a lot of time
model_embeddings = FastTextWrapper.load_fasttext_format(fast_text)
# some initial info about whole dataset
def get_initial_info_data(filepath):
    
    global examples_train
    token_set = set()
    with open(filepath, 'r') as f:
        for line in f:
            #print(line.split(' ')[0].split('#')[0])
            txt = " ".join(line.split(' ')[1:])
            tokens = nltk.word_tokenize(txt)
            examples_train += len(tokens) + 1
            for token in tokens:
                token_set.add(token.lower())
    token_list = list(token_set)
    token_list.insert(0, special_token)
    token_list.append(token_start)
    token_list.append(token_end)
    token_to_int = dict((token, i) for i, token in enumerate(token_list))
    int_to_token = dict((i, token) for i, token in enumerate(token_list))
    return token_to_int, int_to_token, len(token_to_int)

token_to_int, int_to_token, n_vocab = get_initial_info_data(filepath)
print(token_to_int['build'])
print(len(token_to_int))
# print(model_embeddings.wv['take'])


# In[4]:


print(model_embeddings.wv[token_start])
print(model_embeddings.wv[token_end])
print(model_embeddings.wv[special_token])


# In[5]:


# input generator (in batches)
# different batches may have different sizes
# each beach has photos_per_batch photos, each photo with many text samples


# VGG model for image feature extraction
base_model = VGG16(weights='imagenet')
base_model.layers.pop()
print(base_model.summary())
last_layer = base_model.layers[-1].output
img_features = Dense(img_embedding)(last_layer)
vgg_model = Model(inputs = base_model.input, outputs = img_features)

def generate_image_features(img_name):
    # Set up path for the image
    try:
        filename = dir_imgs + '/' + img_name
        # The container of the images (VGG receives 224 x 224 x 3 tensors)
        npix = 224
        target_size = (npix,npix,3)
        data = np.zeros((20,npix,npix,3))
        image = load_img(filename, target_size=target_size)
        image = img_to_array(image)
        nimage = preprocess_input(image)
        #batch size = 1 for now
        batch_size = 1
        y_pred = vgg_model.predict(nimage.reshape( (batch_size,) + nimage.shape[:3]))
    except:
        print('img file not found')
        return None
    
    return y_pred

# Test for image features
generate_image_features("269898095_d00ac7d7a4.jpg")

def generate_text_samples(path, photos_per_batch, n_vocab):
    
    X, y = [], []
    nr_photos = 0
    
    while True:
        with open(path, 'r') as f:
            for line in f:
                nr_photos += 1
                img_name = line.split(' ')[0].split('#')[0]
                img_features = generate_image_features(img_name)
                if img_features is None:
                    continue
                seq = line.split(' ')[1:]
                text = " ".join(seq)
                seq = nltk.word_tokenize(text)
                seq.insert(0, token_start)
                seq.insert(len(seq), token_end)
                seq = [token.lower() for token in seq]
                
                # split one sequence into multiple X, y pairs
                for i in range(1, len(seq)):
                    in_seq_em = []
                    # add padding
                    for j in range(max(0, seq_length - i - 1)):
                        in_seq_em.append(np.float32([0] * img_embedding))
                    # add image
                    in_seq_em.append(img_features[0])
                    # add text
                    for j in range(max(i - seq_length + 1, 0), i):
                        if seq[j] in model_embeddings.wv:
                            in_seq_em.append(model_embeddings.wv[seq[j]])
                        else:
                            in_seq_em.append(model_embeddings.wv[special_token])
                    in_seq_em = np.float32(in_seq_em)
                    try:
                        token_index = token_to_int[seq[i]]
                    except:
                        token_index = 0
                    out_seq = np_utils.to_categorical([token_index], num_classes=n_vocab)[0]
                    X.append(in_seq_em)
                    y.append(out_seq)
                # yield the batch data
                if nr_photos == photos_per_batch:
                    try:
                        X = np.float32(X)
                        X = X.reshape((X.shape[0], seq_length, img_embedding))
                        y = np.float32(y)
                        yield (X, np.float32(y))
                    except:                    
                        print('wrong batch shape')
                    X, y = [], []
                    nr_photos = 0


# In[ ]:





# In[6]:



class GenerateText(Callback):
    # show some generated text at the end of each epoch (start always from <start>)
    def on_epoch_end(self, epoch, logs={}):
        tokens_to_generate = 20
        predictions = []
        img_name = '1305564994_00513f9a5b.jpg'
        Xp = np.zeros((1, seq_length, img_embedding))
        # generate a sequence from starting from start sequence
        for index_sample in range(tokens_to_generate):
            if index_sample == 0:
                start = token_start
                predictions.append(token_to_int[start])
                Xp[0, -2, :] = generate_image_features(img_name)[0]
                # model_embeddings
                Xp[0, -1, :] = np.float32(model_embeddings.wv[start])
                Xp = np.float32(Xp).reshape((1, seq_length, img_embedding))
            else:
                # model_embeddings
                added_word = int_to_token[last_pred]
                if added_word in model_embeddings.wv:    
                    next_token = np.float32(model_embeddings[added_word])
                else:
                    next_token = np.float32(model_embeddings[special_token])
                # eliminate first char, add the last char predicted
                Xp = np.append(Xp[0, 1:, :], next_token).reshape((1, seq_length, img_embedding))
           
            p = self.model.predict(x=Xp, batch_size=None, steps=1)[0]
            last_pred = np.argmax(p)
            predictions.append(last_pred)
            
            if int_to_token[last_pred] == token_end:
                break
        gen_sent = " ".join([int_to_token[p] for p in predictions])
        print('On epoch {} text generated: {}'.format(epoch, gen_sent))
    
#sequential model - not used
# def sequential_model(X, y):
    
#     model = Sequential()
#     model.add(LSTM(32, input_shape=(X.shape[1], X.shape[2])))
#     model.add(Dropout(0.2))
#     model.add(Dense(y.shape[1], activation='softmax'))
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     # define the checkpoint
#     filepath="models/seq-weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
#     checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
#     show_gen_txt = GenerateText()
#     callbacks_list = [checkpoint, show_gen_txt]
#     # fit the model
#     print(model.summary())
#     model.fit_generator(generate_text_samples(filepath, 5, n_vocab),\
#                         epochs=20, batch_size=128, callbacks=callbacks_list, verbose=1)
    
#sequential_model(X, y)


# In[11]:



# functional model 
def functional_model():
    
    inputt = Input(shape=(seq_length, img_embedding))
    char_lstm_last_hidden_state = LSTM(units=lstm_cell,                                       input_shape=(seq_length, img_embedding),                                       return_sequences=False,                                       stateful=False)(inputt)
    # TODO add Input() layer for photos
    output = Dense(n_vocab, activation='softmax')(char_lstm_last_hidden_state)
    model = keras.models.Model(inputs=inputt,                               outputs=output)
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # define the checkpoint
    filepath_save = "models/weights-improvement-{epoch:02d}-{loss:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath_save, monitor='loss', verbose=1, save_best_only=True, mode='min')
    show_gen_txt = GenerateText()
    callbacks_list = [checkpoint, show_gen_txt]
    print(model.summary())
    # fit the model
    model.fit_generator(generate_text_samples(filepath, photos_per_batch, n_vocab),                        epochs=5000, steps_per_epoch=50,                        callbacks=callbacks_list, verbose=1)
    
functional_model()


# In[ ]:





# In[ ]:





# In[ ]:




