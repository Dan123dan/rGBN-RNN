

import gensim.models as g
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import random
import time
import os
import config as cf
import numpy as np
import pickle
import tensorflow as tf
from rGBN_sampler import MyThread
from data_process_all import gen_vocab, gen_data, Get_TM_vocab, Bag_of_words, \
    init_embedding,get_sent_num,get_batch_tm_lm_all,get_doc_sent
import GBN_sampler
from model_rGBN_layer3 import MY_Model

def initialize_Phi_Pi(V_tm):
    Phi = [0] * cf.DPGDS_Layer
    Pi = [0] * cf.DPGDS_Layer
    NDot_Phi = [0] * cf.DPGDS_Layer

    NDot_Pi = [0] * cf.DPGDS_Layer
    for l in range(cf.DPGDS_Layer):
        if l == 0:
            Phi[l] = np.random.rand(V_tm, cf.K[l])
        else:
            Phi[l] = np.random.rand(cf.K[l - 1], cf.K[l])

        Phi[l] = Phi[l] / np.sum(Phi[l], axis=0)
        Pi[l] = np.eye(cf.K[l])
    return Phi, Pi, NDot_Phi, NDot_Pi


def update_Pi_Phi(miniBatch, Phi,Pi, Theta, MBratio, MBObserved,NDot_Phi,NDot_Pi):

    ForgetRate = np.power((cf.Setting['tao0FR'] + np.linspace(1, cf.Setting['Iterall'], cf.Setting['Iterall'])),
                          -cf.Setting['kappa0FR'])
    epsit = np.power((cf.Setting['tao0'] + np.linspace(1, cf.Setting['Iterall'], cf.Setting['Iterall'])), -cf.Setting['kappa0'])
    epsit = cf.Setting['epsi0'] * epsit / epsit[0]

    L = cf.DPGDS_Layer
    A_VK = [0]* L
    L_KK = [0]* L
    Piprior = [0]* L
    EWSZS_Phi = [0]* L
    EWSZS_Pi = [0]* L

    Xi = []
    Vk = []
    for l in range(L):
        Xi.append(1)
        Vk.append(np.ones((cf.K[l], 1)))

    threads = []

    for i in range(cf.Batch_Size):
        Theta1 = Theta[0][ i, :, :]
        Theta2 = Theta[1][ i, :, :]
        Theta3 = Theta[2][ i, :, :]

        t = MyThread(i, np.transpose(miniBatch[i, :, :]), Phi, Theta1, Theta2, Theta3, L, cf.K , cf.sent_J, Pi)
        threads.append(t)
    for t in threads:
        t.setDaemon(True)
        t.start()
    for t in threads:
        t.join()
    for t in threads:
        AA, BB, CC = t.get_result()
        for l in range(L):
            A_VK[l] = A_VK[l] + BB[l]
            L_KK[l] = L_KK[l] + CC[l]

    for l in range(len(Phi)):
        EWSZS_Phi[l] = MBratio * A_VK[l]
        EWSZS_Pi[l] = MBratio * L_KK[l]

        if (MBObserved == 0):
            NDot_Phi[l] = EWSZS_Phi[l].sum(0)
            NDot_Pi[l] = EWSZS_Pi[l].sum(0)
        else:
            NDot_Phi[l] = (1 - ForgetRate[MBObserved]) * NDot_Phi[l] + ForgetRate[MBObserved] * EWSZS_Phi[l].sum(
                0) 
            NDot_Pi[l] = (1 - ForgetRate[MBObserved]) * NDot_Pi[l] + ForgetRate[MBObserved] * EWSZS_Pi[l].sum(0)  

        tmp = EWSZS_Phi[l] + cf.eta0  
        tmp = (1 / np.maximum(NDot_Phi[l], cf.real_min)) * (tmp - tmp.sum(0) * Phi[l])  
        tmp1 = (2 / np.maximum(NDot_Phi[l], cf.real_min)) * Phi[l]
        tmp = Phi[l] + epsit[MBObserved] * tmp + np.sqrt(epsit[MBObserved] * tmp1) * np.random.randn(Phi[l].shape[0],Phi[l].shape[1])
        Phi[l] = GBN_sampler.ProjSimplexSpecial(tmp, Phi[l], 0)

        Piprior[l] = np.dot(Vk[l], np.transpose(Vk[l]))
        Piprior[l][np.arange(Piprior[l].shape[0]), np.arange(Piprior[l].shape[1])] = 0
        Piprior[l] = Piprior[l] + np.diag(np.reshape(Xi[l] * Vk[l], Vk[l].shape[0], 1))

        tmp = EWSZS_Pi[l] + Piprior[l]  
        tmp = (1 / np.maximum(NDot_Pi[l], cf.real_min)) * (tmp - tmp.sum(0) * Pi[l])  
        tmp1 = (2 / np.maximum(NDot_Pi[l], cf.real_min)) * Pi[l]
        tmp = Pi[l] + epsit[MBObserved] * tmp + np.sqrt(epsit[MBObserved] * tmp1) * np.random.randn(Pi[l].shape[0],
                                                                                                    Pi[l].shape[1])
        Pi[l] = GBN_sampler.ProjSimplexSpecial(tmp, Pi[l], 0)

    return Phi, Pi, NDot_Phi, NDot_Pi

def log_max(input_x):
    return tf.log(tf.maximum(input_x, cf.real_min))


# set the seeds
random.seed(cf.seed)
np.random.seed(cf.seed)

# globals
vocabxid = {}
idxvocab = []

# constants
pad_symbol = "<pad>"
start_symbol = "<go>"
end_symbol = "<eos>"
unk_symbol = "<unk>"
dummy_symbols = [pad_symbol, start_symbol, end_symbol, unk_symbol]

print("First pass on train corpus to collect vocabulary stats...")
idxvocab, vocabxid, tm_ignore = gen_vocab(dummy_symbols, cf.train_corpus, cf.stopwords, cf.vocab_minfreq,
                                           cf.vocab_maxfreq, cf.verbose)
#
print("Processing train corpus to collect sentence and document data...")
train_sents, train_docs, train_docids, train_stats = gen_data(vocabxid, dummy_symbols, tm_ignore, cf.train_corpus,
                                                              cf.tm_sent_len, cf.lm_sent_len, cf.verbose, False)

TM_vocab = Get_TM_vocab(idxvocab, tm_ignore)


if cf.dataname == 'apnews' or cf.dataname == 'bnc':
    TM_train_doc, train_doc_bow = Bag_of_words(train_docs[0], idxvocab, tm_ignore)
elif cf.dataname == 'imdb':  # imdb traindata is too big to save
    TM_train_doc, train_doc_bow = Bag_of_words(train_docs[0], idxvocab, tm_ignore)
else:
    print("There is another dataset ! ")

print('-----------------------------data ----------load---------------------------------------- ')

################################################################### data prepare ##############################################################################
train_sent_num = get_sent_num(train_sents[1],len(train_doc_bow[0]))
train_Doc = get_doc_sent(train_sents[1], train_doc_bow, train_sent_num, cf.sent_J)

doc_num_batches = int(np.floor(float(len(train_Doc)) / cf.Batch_Size))
batch_ids = [item for item in range(doc_num_batches)]


V_tm = len(idxvocab) - len(tm_ignore)
Phi, Pi, NDot_Phi, NDot_Pi = initialize_Phi_Pi(V_tm)


cf.Setting['Iterall'] = cf.epoch_size * doc_num_batches



if cf.save_model:
    if not os.path.exists(os.path.join(cf.output_dir, cf.output_prefix)):
        os.makedirs(os.path.join(cf.output_dir, cf.output_prefix))
    if not os.path.exists(os.path.join(cf.output_dir, cf.output_path)):
        os.makedirs(os.path.join(cf.output_dir, cf.output_path))

graph = tf.get_default_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with graph.as_default(), tf.Session(config = config) as sess:
    tf.set_random_seed(cf.seed)
    DL = MY_Model(cf, V_tm, len(vocabxid))
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.all_variables(), max_to_keep=0)

    if os.path.exists(cf.word_embedding_model):
        print("Loading word embedding model...")
        # mword = g.Word2Vec.load(cf.word_embedding_model)
        mword = g.KeyedVectors.load_word2vec_format(cf.word_embedding_model, binary=True)
        cf.word_embedding_size = mword.vector_size
        word_emb = init_embedding(mword, idxvocab)  # mword is a word embedding from gensim.wordembedding
        sess.run(DL.LSTM_word_embedding.assign(word_emb))

    test_likelihood = []
    train_likelihood = []

    for e in range(cf.tm_epoch_size):
        train_theta = []
        print("\nEpoch =", e)
        time_start = time.time()
        random.shuffle(batch_ids)

        for batch_id in batch_ids:
            MBObserved = int(e * doc_num_batches + batch_id)
            Doc = train_Doc[(batch_id * cf.Batch_Size):((batch_id + 1) * cf.Batch_Size)]
            X_train_batch, y_train_batch,d_train_batch, m_train_batch = get_batch_tm_lm_all(Doc,train_doc_bow,len(idxvocab),tm_ignore,cf.Batch_Size)

            _, tm_cost,train_like, Theta = sess.run([DL.tm_train_step,DL.tm_Loss, DL.L1,[DL.theta_1C_HT,DL.theta_2C_HT,DL.theta_3C_HT]],
                                    feed_dict={DL.Doc_input:d_train_batch, DL.phi_1:Phi[0], DL.phi_2:Phi[1],DL.phi_3:Phi[2],
                                               DL.pi_1:Pi[0], DL.pi_2:Pi[1] , DL.pi_3:Pi[2],  DL.droprate:cf.lm_keep_prob })
            Phi, Pi, NDot_Phi, NDot_Pi = update_Pi_Phi(d_train_batch, Phi, Pi, Theta, doc_num_batches, MBObserved, NDot_Phi, NDot_Pi)

            train_theta.append(Theta) 
            train_likelihood.append(train_like)

            if MBObserved%100 == 0:
                print("\nMinibatch =", batch_id)
                print("\ntopic model Cost:", tm_cost)
                print("\ntopic model Cost:", train_like)

            if cf.save_model and MBObserved%1000==0:

                print('----------------------------------- saving model -------------------------------------')
                saver.save(sess, os.path.join(cf.output_dir, cf.output_prefix, cf.dataname + str(e) + "_model_" + str(batch_id) + ".ckpt"))
                Phi_Pi_save = open(os.path.join(cf.output_dir, cf.output_path, cf.dataname + str(e) + '_Phi_Pi_hidden_Doc_' + str(batch_id) + '.pckl'), 'wb')
                pickle.dump([Phi, Pi], Phi_Pi_save)
                Phi_Pi_save.close()
