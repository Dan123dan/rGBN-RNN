Code for a three-layer rGBN-RNN model proposed in Recurrent Hierarchical Topic-Guided RNN for Language Generation,  ICML 2020.

Dependencies
This code is written in python. To use it you will need:
Python 3.5+
Tensorflow-gpu 1.9.0
A recent version of NumPy and SciPy
pickle
gensim

Getting started:
1. Download datasets from the links presented in appendixe C of the paper, and put into the data directory. 
2. Download the pre-trained 300-dimension word2vec Google News vectors, and put into the data directory.
3. Train the rGBN-RNN model using 'train_rGBN.py' , the data-preprocessed code is included in this python file.

Data download link：
appnews, imdb, bnc: https://ibm.ent.box.com/s/ls61p8ovc1y87w45oa02zink2zl7l6z4
word2vec Google News embedding: https://code.google.com/archive/p/word2vec/
The stoplist is concluded in our project, named as 'en.txt'.

Note:
We provid a short text 'example.txt' from the traning text of appnews dataset, which can be directly run using our code.
If you run this code in Windows system , you may need to install visual studio firstly , and change '***.so'  to  '***.dll'  files in  'GBN_sampler.py'.



