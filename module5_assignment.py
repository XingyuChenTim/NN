import string
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Model
from keras.layers import LSTM, Input, TimeDistributed, Dense, Activation, RepeatVector, Embedding
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy

# GOAL--------------------------------------------------------
# Split model into two parts,
# An encoder that inputs the Spanish sentence and produces a hidden vector.
# The encoder is built with an Embedding layer that converts the words
# into a vector and a recurrent neural network (RNN) that calculates the hidden state,
# here we will be using Long Short-Term Memory (LSTM) layer.

# Then the output of the encoder will be used as input for the decoder.
# For the decoder, we will be using LSTM layer again, as well as a dense layer
# that predicts the English word.
# -------------------------------------------------------------------


# Read file
filename = "fra.txt"
translation_file = open(filename, "r", encoding='utf-8')
raw_data = translation_file.read()
translation_file.close()

# Parse data
raw_data = raw_data.split('\n')
print(raw_data[1000])
pairs = [sentence.split('\t') for sentence in raw_data]
print(pairs[1000])
pairs = pairs[1000:20000]


###############################
# Pre-process the data and get the maximum length of Spanish and English sentences.
##############################
def clean_sentence(sentence):
    # Lower case the sentence
    lower_case_sent = sentence.lower()
    # Strip punctuation
    string_punctuation = string.punctuation + "¡" + "¿" + "?"
    clean_sentence = lower_case_sent.translate(str.maketrans("", "", string_punctuation))
    return clean_sentence


def tokenize(sentences):
    # Create tokenizer
    text_tokenizer = Tokenizer()
    # Fit texts
    text_tokenizer.fit_on_texts(sentences)
    return text_tokenizer.texts_to_sequences(sentences), text_tokenizer


# Clean sentences
english_sentences = [clean_sentence(pair[0]) for pair in pairs]
spanish_sentences = [clean_sentence(pair[1]) for pair in pairs]

# Test
print(english_sentences[1000])
tt, st = tokenize(english_sentences)
print(tt[1000])
print(st)

spa_text_tokenized, spa_text_tokenizer = tokenize(spanish_sentences)
print(spa_text_tokenized[0:5], spa_text_tokenizer)
eng_text_tokenized, eng_text_tokenizer = tokenize(english_sentences)

print('Maximum length spanish sentence: {}'.format(len(max(spa_text_tokenized, key=len))))
print('Maximum length english sentence: {}'.format(len(max(eng_text_tokenized, key=len))))

##############################################
# We apply padding to make the maximum length of the sentences in each language equal.
################################################################################
# Check language length
spanish_vocab = len(spa_text_tokenizer.word_index) + 1
english_vocab = len(eng_text_tokenizer.word_index) + 1
print("Spanish vocabulary is of {} unique words".format(spanish_vocab))
print("English vocabulary is of {} unique words".format(english_vocab))

max_spanish_len = int(len(max(spa_text_tokenized, key=len)))
max_english_len = int(len(max(eng_text_tokenized, key=len)))

spa_pad_sentence = pad_sequences(spa_text_tokenized, max_spanish_len, padding="post")
eng_pad_sentence = pad_sequences(eng_text_tokenized, max_english_len, padding="post")

# Reshape data
spa_pad_sentence = spa_pad_sentence.reshape(*spa_pad_sentence.shape, 1)
eng_pad_sentence = eng_pad_sentence.reshape(*eng_pad_sentence.shape, 1)

#####################################################
##
#  Build the model
##
#####################################################

# Last Layer
# We have one last step, to predict the translated word.
# For this we need to use a Dense Layer. The parameter
# we need to define is the number of units, this number
# of units is the shape of the output vector and it
# needs to be the same as the length of the English vocabulary.
# Why? The vector will be all values close to zero, except
# one of the units that will be close to 1. We then need
# to map the index of the unit that outputs a 1 with a
# dictionary where we map each unit to a word. For example,
# if the input is the word ‘sol’ and the output is a vector
# where all are zeros and then the unit 472 is 1, we map
# this index against the dictionary containing the English
# words and we get the value ‘sun’.
input_sequence = Input(shape=(max_spanish_len,))
embedding = Embedding(input_dim=spanish_vocab, output_dim=128, )(input_sequence)
print(embedding.shape)
encoder = LSTM(64, return_sequences=False)(embedding)
r_vec = RepeatVector(max_english_len)(encoder)
decoder = LSTM(64, return_sequences=True, dropout=0.2)(r_vec)
logits = TimeDistributed(Dense(english_vocab))(decoder)

# Stack the layers to create the model and add a function loss Compile
enc_dec_model = Model(input_sequence, Activation('softmax')(logits))
enc_dec_model.compile(loss=sparse_categorical_crossentropy,
                      optimizer=Adam(1e-3),
                      metrics=['accuracy'])
enc_dec_model.summary()

# Train the model
model_results = enc_dec_model.fit(spa_pad_sentence, eng_pad_sentence, batch_size=30, epochs=100)


# When the model is trained we can make our first translation.
# You will also find the function ‘logits_to_sentence’
# that maps the output of the dense layer with the English vocabulary.

def logits_to_sentence(logits, tokenizer):
    index_to_words = {idx: word for word, idx in tokenizer.word_index.items()}
    index_to_words[0] = '<empty>'

    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])


index = 1000
print("The english sentence is: {}".format(english_sentences[index]))
print("The spanish sentence is: {}".format(spanish_sentences[index]))
print('The predicted sentence is :')
print(logits_to_sentence(enc_dec_model.predict(spa_pad_sentence[index:index + 1])[0], eng_text_tokenizer))
