#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 22:21:44 2018

@author: cb413
"""

import numpy as np
import tensorflow as tf
import pickle

class MY_Model():
    def __init__(self, config, V_tm, V_lm):

#######################################DPGDS_PARAM###############################################
        # self.ForgetRate = np.power((config.Setting['tao0FR'] + np.linspace(1, config.Setting['Iterall'], config.Setting['Iterall'])),
        #                       -config.Setting['kappa0FR'])
        # epsit = np.power((config.Setting['tao0'] + np.linspace(1, config.Setting['Iterall'], config.Setting['Iterall'])), -config.Setting['kappa0'])
        # self.epsit = config.Setting['epsi0'] * epsit / epsit[0]

        # params
        self.V_dim = V_tm  ## DPGDS vocabulary lenth
        self.H_dim = config.H     ## DPGDS encoder dimension
        self.K_dim = config.K     ## DPGDS topic dimension
        self.Batch_Size = config.Batch_Size
        self.L = config.DPGDS_Layer   ## DPGDS layer
        self.real_min = config.real_min

        self.seed = config.seed
        self.W_1, self.W_2, self.W_3, self.W_4, self.W_5, self.W_6, self.W_1_k, self.b_1_k, \
        self.W_1_l, self.b_1_l, self.W_3_k, self.b_3_k, self.W_3_l, self.b_3_l, self.W_5_k, \
        self.b_5_k, self.W_5_l, self.b_5_l = self.initialize_weight()

        self.Doc_input = tf.placeholder("float32", shape=[self.Batch_Size, config.sent_J, self.V_dim])  # N*J*V
        self.phi_1 = tf.placeholder(tf.float32, shape=[self.V_dim, self.K_dim[0]])  # N*V
        self.phi_2 = tf.placeholder(tf.float32, shape=[self.K_dim[0], self.K_dim[1]])  # N*V
        self.phi_3 = tf.placeholder(tf.float32, shape=[self.K_dim[1], self.K_dim[2]])  # N*V
        self.pi_1 = tf.placeholder(tf.float32, shape=[self.K_dim[0], self.K_dim[0]])  # N*V
        self.pi_2 = tf.placeholder(tf.float32, shape=[self.K_dim[1], self.K_dim[1]])  # N*V
        self.pi_3 = tf.placeholder(tf.float32, shape=[self.K_dim[2], self.K_dim[2]])  # N*V


        self.state1 = tf.zeros([self.Batch_Size, self.H_dim[0]], dtype=tf.float32)
        self.state2 = tf.zeros([self.Batch_Size, self.H_dim[1]], dtype=tf.float32)
        self.state3 = tf.zeros([self.Batch_Size, self.H_dim[2]], dtype=tf.float32)

        self.theta_1C_HT = [];  self.theta_2C_HT = [];  self.theta_3C_HT = []
        self.h1 = [];    self.h2 = [];    self.h3 = []
        self.LB = 0;  self.L1 = 0;

#######################################LSTM_PARAM###############################################

        self.Sent_input = tf.placeholder(tf.int32, [self.Batch_Size, config.sent_J, config.lm_sent_len])
        self.lm_mask = tf.placeholder(tf.float32,  [self.Batch_Size, config.sent_J, config.lm_sent_len])
        self.Sent_output = tf.placeholder(tf.int32,[self.Batch_Size, config.sent_J, config.lm_sent_len])

        self.droprate = tf.placeholder(tf.float32,[])

        self.vocab_size = V_lm
        self.LSTM_word_embedding = tf.get_variable("lstm_embedding", [self.vocab_size, config.word_embedding_size],
                                              trainable=config.word_embedding_update,
                                              initializer=tf.random_uniform_initializer(-0.5 / config.word_embedding_size,
                                                                                        0.5 / config.word_embedding_size))
        self.lstm_inputs = tf.nn.embedding_lookup(self.LSTM_word_embedding, self.Sent_input)
        self.lstm_inputs = tf.nn.dropout(self.lstm_inputs, self.droprate, seed=self.seed)

        self.hidden1 = []
        self.hidden2 = []
        self.hidden3 = []
        self.cell_1 = tf.nn.rnn_cell.BasicLSTMCell(config.rnn_hidden_size1)
        self.cell_2 = tf.nn.rnn_cell.BasicLSTMCell(config.rnn_hidden_size2)
        self.cell_3 = tf.nn.rnn_cell.BasicLSTMCell(config.rnn_hidden_size3)

        self.Uz1, self.Ur1, self.Uh1, self.Wz1, self.Wr1, self.Wh1, self.bz1, self.br1, self.bh1 = self.GRU_params(config.theta_size1,config.rnn_hidden_size1,config.rnn_bias)
        self.Uz2, self.Ur2, self.Uh2, self.Wz2, self.Wr2, self.Wh2, self.bz2, self.br2, self.bh2 = self.GRU_params(config.theta_size2,config.rnn_hidden_size2,config.rnn_bias)
        self.Uz3, self.Ur3, self.Uh3, self.Wz3, self.Wr3, self.Wh3, self.bz3, self.br3, self.bh3 = self.GRU_params(config.theta_size3,config.rnn_hidden_size3,config.rnn_bias)


        self.batch_num = tf.placeholder(tf.int32, [])


        for j in range(config.sent_J):
            input_X = self.Doc_input[:,j,:]  ### N*V
            ##################  DPGDS layer1  ########################
            self.state1 = tf.sigmoid(tf.matmul(input_X, self.W_1))

            self.k_1, self.l_1 = self.Encoder_Weilbull(self.state1, 0, self.W_1_k, self.b_1_k, self.W_1_l, self.b_1_l)  # K*N,  K*N
            theta_1, theta_1c = self.reparameterization(self.k_1, self.l_1, 0, self.Batch_Size)  # K * N batch_size = 20
            ##################  DPGDS layer2  ########################
            self.state2 = tf.sigmoid(tf.matmul(self.state1, self.W_3) )
                # state2 = tf.zeros([self.Batch_Size, self.H_dim[1]], dtype=tf.float32)
            # self.state2 = tf.sigmoid(tf.matmul(self.state1, self.W_3) + tf.matmul(self.state2, self.W_4))
            self.k_2, self.l_2 = self.Encoder_Weilbull(self.state2, 1, self.W_3_k, self.b_3_k, self.W_3_l, self.b_3_l)
            theta_2, theta_2c = self.reparameterization(self.k_2, self.l_2, 1, self.Batch_Size)
            ##################  DPGDS layer3  ########################
            self.state3 = tf.sigmoid(tf.matmul(self.state2, self.W_5))
                # self.state3 = tf.zeros([self.Batch_Size, self.H_dim[2]], dtype=tf.float32)
            # self.state3 = tf.sigmoid(tf.matmul(self.state2, self.W_5) + tf.matmul(self.state3, self.W_6))
            self.k_3, self.l_3 = self.Encoder_Weilbull(self.state3, 2, self.W_5_k, self.b_5_k, self.W_5_l, self.b_5_l)
            theta_3, theta_3c = self.reparameterization(self.k_3, self.l_3, 2, self.Batch_Size)

            self.h1.append(self.state1)
            self.h2.append(self.state2)
            self.h3.append(self.state3)
            if j == 0:
                alpha_1_t = tf.matmul(self.phi_2, theta_2)
                alpha_2_t = tf.matmul(self.phi_3, theta_3)
                alpha_3_t = tf.ones([self.K_dim[2], self.Batch_Size], dtype='float32')  # K * 1
            else:
                # if self.is_training is False:
                #     alpha_1_t = tf.matmul(self.phi_2, theta_2)
                #     alpha_2_t = tf.matmul(self.phi_3, theta_3)
                #     alpha_3_t = tf.ones([self.K_dim[2], self.Batch_Size], dtype='float32')  # K * 1
                # else:
                alpha_1_t = tf.matmul(self.phi_2, theta_2) + tf.matmul(self.pi_1, theta_left_1)
                alpha_2_t = tf.matmul(self.phi_3, theta_3) + tf.matmul(self.pi_2, theta_left_2)
                alpha_3_t = tf.matmul(self.pi_3, theta_left_3)

            L1_1_t = (tf.transpose(input_X)) * self.log_max_tf(tf.matmul(self.phi_1, theta_1)) - tf.matmul(self.phi_1, theta_1)  # - tf.lgamma( X_VN_t + 1)
            theta1_KL = tf.reduce_sum(self.KL_GamWei(alpha_1_t, np.float32(1.0), self.k_1, self.l_1))
            theta2_KL = tf.reduce_sum(self.KL_GamWei(alpha_2_t, np.float32(1.0), self.k_2, self.l_2))
            theta3_KL = tf.reduce_sum(self.KL_GamWei(alpha_3_t, np.float32(1.0), self.k_3, self.l_3))

            self.LB = self.LB + (1 * tf.reduce_sum(L1_1_t) + 0.1 * theta1_KL + 0.01 * theta2_KL + 0.001 * theta3_KL)/self.Batch_Size
            self.L1 = self.L1 + tf.reduce_sum(L1_1_t)/self.Batch_Size

            theta_left_1 = theta_1
            theta_left_2 = theta_2
            theta_left_3 = theta_3


            self.theta_1C_HT.append(theta_1c)
            self.theta_2C_HT.append(theta_2c)
            self.theta_3C_HT.append(theta_3c)


            self.theta_1c_norm = theta_1c/tf.maximum(tf.tile(tf.reshape(tf.reduce_max(theta_1c,axis=1),[self.Batch_Size,1]),[1,self.K_dim[0]]), config.real_min)
            self.theta_2c_norm = theta_2c/tf.maximum(tf.tile(tf.reshape(tf.reduce_max(theta_2c,axis=1),[self.Batch_Size,1]),[1,self.K_dim[1]]), config.real_min)
            self.theta_3c_norm = theta_3c/tf.maximum(tf.tile(tf.reshape(tf.reduce_max(theta_3c,axis=1),[self.Batch_Size,1]),[1,self.K_dim[2]]), config.real_min)


            self.initial_state_t_1 = self.cell_1.zero_state(config.Batch_Size, tf.float32)
            self.initial_state_t_2 = self.cell_2.zero_state(config.Batch_Size, tf.float32)
            self.initial_state_t_3 = self.cell_3.zero_state(config.Batch_Size, tf.float32)

            for t in range(config.lm_sent_len):
                x_t = self.lstm_inputs[:, j, t, :]
                ##################  LSTM layer1  ########################
                if t==0:
                    self.H_t_1,self.state_t_1 = self.cell_1(x_t,self.initial_state_t_1)
                else:
                    self.H_t_1, self.state_t_1 = self.cell_1(x_t,self.state_t_1)
                self.h_t_1_theta = self.GRU_theta_hidden(self.theta_1c_norm, self.H_t_1, self.Uz1, self.Ur1, self.Uh1, self.Wz1, self.Wr1, self.Wh1, self.bz1, self.br1, self.bh1)
                h_t_1_drop = tf.nn.dropout(self.h_t_1_theta, self.droprate, seed=self.seed )
                self.hidden1.append(h_t_1_drop)
                ##################  LSTM layer2  ########################
                if t==0:
                    self.H_t_2, self.state_t_2 = self.cell_2(h_t_1_drop, self.initial_state_t_2)
                else:
                    self.H_t_2, self.state_t_2 = self.cell_2(h_t_1_drop, self.state_t_2)
                self.h_t_2_theta = self.GRU_theta_hidden(self.theta_2c_norm, self.H_t_2, self.Uz2, self.Ur2, self.Uh2, self.Wz2, self.Wr2, self.Wh2, self.bz2, self.br2, self.bh2 )
                h_t_2_drop = tf.nn.dropout(self.h_t_2_theta, self.droprate, seed=self.seed)
                self.hidden2.append(h_t_2_drop)
                ##################  LSTM layer3  ########################
                if t==0:
                    self.H_t_3, self.state_t_3 = self.cell_3(h_t_2_drop, self.initial_state_t_3)
                else:
                    self.H_t_3, self.state_t_3 = self.cell_3(h_t_2_drop, self.state_t_3)

                self.h_t_3_theta = self.GRU_theta_hidden(self.theta_3c_norm, self.H_t_3, self.Uz3, self.Ur3, self.Uh3, self.Wz3, self.Wr3, self.Wh3, self.bz3, self.br3, self.bh3 )
                h_t_3_drop = tf.nn.dropout(self.h_t_3_theta, self.droprate, seed=self.seed)
                self.hidden3.append(h_t_3_drop)

        self.theta_1C_HT = tf.transpose(self.theta_1C_HT,[1,2,0])
        self.theta_2C_HT = tf.transpose(self.theta_2C_HT,[1,2,0])
        self.theta_3C_HT = tf.transpose(self.theta_3C_HT,[1,2,0])

        hidden_all = tf.concat([self.hidden1, self.hidden2, self.hidden3],-1)
        self.hidden = tf.transpose(hidden_all,perm=[1, 0, 2])
        hidden_size = config.rnn_hidden_size1 + config.rnn_hidden_size2 + config.rnn_hidden_size3
        self.lm_softmax_w = tf.get_variable("lm_softmax_w", [hidden_size, self.vocab_size])
        self.lm_softmax_b = tf.get_variable("lm_softmax_b", [self.vocab_size], initializer=tf.constant_initializer())
        self.lm_logits = tf.matmul(tf.reshape(self.hidden, [-1, hidden_size]), self.lm_softmax_w) + self.lm_softmax_b

        self.tm_Loss = - self.LB
        self.tm_train_step = tf.train.AdamOptimizer(config.learning_rate).minimize(self.tm_Loss)

        lm_crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(self.Sent_output, [-1]), logits=self.lm_logits)
        lm_crossent_m = lm_crossent * tf.reshape(self.lm_mask, [-1])
        self.lm_Loss = tf.reduce_sum(lm_crossent_m) / self.Batch_Size

        self.joint_Loss = 0.001*self.tm_Loss + self.lm_Loss

 
        learning_rate = tf.train.exponential_decay(config.learning_rate,self.batch_num,5000,0.98)
        Optimizer = tf.train.AdamOptimizer(learning_rate)
        threshold = 1
        grads_vars = Optimizer.compute_gradients(self.joint_Loss)
        capped_gvs = []
        for grad, var in grads_vars:
            if grad is not None:
                grad = tf.where(tf.is_nan(grad), threshold * tf.ones_like(grad), grad)
                grad = tf.where(tf.is_inf(grad), threshold * tf.ones_like(grad), grad)
                capped_gvs.append((tf.clip_by_value(grad, -threshold, threshold), var))
        self.joint_train_step = Optimizer.apply_gradients(capped_gvs)




    def GRU_params(self,input_size,output_size,rnn_bias):
        Uz = self.rnn_weight_variable([input_size, output_size])
        Ur = self.rnn_weight_variable([input_size, output_size])
        Uh = self.rnn_weight_variable([input_size, output_size])
        Wz = self.rnn_weight_variable([output_size, output_size])
        Wr = self.rnn_weight_variable([output_size, output_size])
        Wh = self.rnn_weight_variable([output_size, output_size])
        bz = self.rnn_bias_variable(rnn_bias, [output_size, ])
        br = self.rnn_bias_variable(rnn_bias, [output_size, ])
        bh = self.rnn_bias_variable(rnn_bias, [output_size, ])
        return Uz,Ur,Uh,Wz,Wr,Wh,bz,br,bh
    def GRU_theta_hidden(self,t_t,h_t_1,Uz,Ur,Uh,Wz,Wr,Wh,bz,br,bh):
        z = tf.nn.sigmoid(tf.matmul(t_t, Uz) + tf.matmul(h_t_1, Wz) + bz)
        r = tf.nn.sigmoid(tf.matmul(t_t, Ur) + tf.matmul(h_t_1, Wr) + br)
        h = tf.nn.tanh(tf.matmul(t_t, Uh) + tf.matmul(tf.multiply(h_t_1, r), Wh)  + bh)
        s_t = tf.multiply(1 - z, h) + tf.multiply(z, h_t_1)
        return s_t


    def weight_variable(self,shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.01, dtype=tf.float32))


    def bias_variable(self,shape):
        return tf.Variable(tf.constant(0.1, shape=shape, dtype=tf.float32))


    def reparameterization(self, Wei_shape, Wei_scale, l, Batch_Size):
        eps = tf.random_uniform(shape=[np.int32(self.K_dim[l]), Batch_Size], dtype=tf.float32)  # K_dim[i] * none
        # eps = tf.ones(shape=[np.int32(self.K_dim[l]), Batch_Size], dtype=tf.float32) /2 # K_dim[i] * none
        theta = Wei_scale * tf.pow(-self.log_max_tf(1 - eps), 1 / Wei_shape)
        theta_c = tf.transpose(theta)
        return theta, theta_c  # K*N    N*K

    def Encoder_Weilbull(self,input_x, l, W_k, b_k, W_l, b_l):  # i = 0:T-1 , input_x N*V
        # feedforward
        k_tmp = tf.nn.softplus(tf.matmul(input_x, W_k) + b_k)  # none * 1
        k_tmp = tf.tile(k_tmp,
                        [1, self.K_dim[l]])  # reshpe   ????                                             # none * K_dim[i]
        k = tf.maximum(k_tmp, self.real_min)
        lam = tf.nn.softplus(tf.matmul(input_x, W_l) + b_l)  # none * K_dim[i]
        return tf.transpose(k), tf.transpose(lam)


    def log_max_tf(self,input_x):
        return tf.log(tf.maximum(input_x, self.real_min))


    def KL_GamWei(self,Gam_shape, Gam_scale, Wei_shape, Wei_scale):  # K_dim[i] * none
        eulergamma = 0.5772
        KL_Part1 = eulergamma * (1 - 1 / Wei_shape) + self.log_max_tf(Wei_scale / Wei_shape) + 1 + Gam_shape * self.log_max_tf(
            Gam_scale)
        KL_Part2 = -tf.lgamma(Gam_shape) + (Gam_shape - 1) * (self.log_max_tf(Wei_scale) - eulergamma / Wei_shape)
        KL = KL_Part1 + KL_Part2 - Gam_scale * Wei_scale * tf.exp(tf.lgamma(1 + 1 / Wei_shape))
        return KL


    def initialize_weight(self):
        V_dim = self.V_dim
        H_dim = self.H_dim
        K_dim = self.K_dim
        W_1 = self.weight_variable(shape=[V_dim, H_dim[0]])
        W_2 = tf.Variable(tf.eye(H_dim[0], dtype=tf.float32))
        W_3 = self.weight_variable(shape=[H_dim[0], H_dim[1]])
        W_4 = tf.Variable(tf.eye(H_dim[1], dtype=tf.float32))
        W_5 = self.weight_variable(shape=[H_dim[1], H_dim[2]])
        W_6 = tf.Variable(tf.eye(H_dim[2], dtype=tf.float32))
        W_1_k = self.weight_variable(shape=[H_dim[0], 1])
        b_1_k = self.bias_variable(shape=[1])
        W_1_l = self.weight_variable(shape=[H_dim[0], K_dim[0]])
        b_1_l = self.bias_variable(shape=[K_dim[0]])
        W_3_k = self.weight_variable(shape=[H_dim[1], 1])
        b_3_k = self.bias_variable(shape=[1])
        W_3_l = self.weight_variable(shape=[H_dim[1], K_dim[1]])
        b_3_l = self.bias_variable(shape=[K_dim[1]])
        W_5_k = self.weight_variable(shape=[H_dim[2], 1])
        b_5_k = self.bias_variable(shape=[1])
        W_5_l = self.weight_variable(shape=[H_dim[2], K_dim[2]])
        b_5_l = self.bias_variable(shape=[K_dim[2]])
        return W_1, W_2, W_3, W_4, W_5, W_6, W_1_k, b_1_k, W_1_l, b_1_l, W_3_k, b_3_k, W_3_l, b_3_l, W_5_k, b_5_k, W_5_l, b_5_l



    def rnn_weight_variable(self,shape):
        return tf.Variable(tf.random_uniform(shape, minval=-0.05, maxval=0.05, seed = self.seed, dtype=tf.float32),
                           trainable=True)


    def rnn_bias_variable(self, bias, shape):
        return tf.Variable(tf.constant(bias, shape=shape, dtype=tf.float32), trainable=True)

    def softmax(self,x):
        x_exp = np.exp(x)
        # 如果是列向量，则axis=0
        x_sum = np.sum(x_exp, keepdims=True)
        s = x_exp / x_sum
        return s

    def sample(self, probs, temperature):
        if temperature == 0:
            return np.argmax(probs)
        probs = probs.astype(np.float64) #convert to float64 for higher precision
        probs = np.log(probs) / temperature
        probs = np.exp(probs) / np.sum(np.exp(probs))
        probs = np.reshape(probs,[probs.shape[1],])

        return np.argmax(np.random.multinomial(1, probs, 1))


    #generate a sentence given conv_hidden
    def generate(self, sess,config, Theta, start_word_id, temperature, max_length, stop_word_id):
        state_t_1 = sess.run(self.cell_1.zero_state(1, tf.float32))
        state_t_2 = sess.run(self.cell_2.zero_state(1, tf.float32))
        state_t_3 = sess.run(self.cell_3.zero_state(1, tf.float32))
        # state_t_1 = (np.ones(1, config.rnn_hidden_size1),np.ones(1, config.rnn_hidden_size1))
        # state_t_2 = (np.ones(1, config.rnn_hidden_size2),np.ones(1, config.rnn_hidden_size2))
        # state_t_3 = (np.ones(1, config.rnn_hidden_size3),np.ones(1, config.rnn_hidden_size3))
        x = [[start_word_id]]
        sent = [start_word_id]
        theta1 = 1*np.reshape(Theta[0],[1,config.K[0]])
        theta2 = 1*np.reshape(Theta[1],[1,config.K[1]])
        theta3 = 1*np.reshape(Theta[2],[1,config.K[2]])
        for _ in range(max_length):
            lm_logits, state_t_1, state_t_2,  state_t_3= sess.run([self.lm_logits, self.state_t_1, self.state_t_2, self.state_t_3],
                    {self.Sent_input: np.reshape(x,[1,1,1]), self.initial_state_t_1: state_t_1, self.initial_state_t_2: state_t_2,
                     self.initial_state_t_3: state_t_3, self.theta_1c: theta1, self.theta_2c: theta2, self.theta_3c: theta3})

            lm_logits_exp = np.exp(lm_logits)
            lm_logits_sum = np.sum(lm_logits_exp, keepdims=True)
            probs = lm_logits_exp / lm_logits_sum
            sent.append(self.sample(probs, temperature))
            if sent[-1] == stop_word_id:
                break
            x = [[ sent[-1] ]]
        return sent

    #generate a sequence of words, given a topic
    def generate_on_topic(self, sess, config, Phi, topic_id, start_word_id, temperature=0.75, max_length=30, stop_word_id=None):
        index = topic_id
        theta = []
        ind = []
        for layer in range(2, -1, -1):
            thet = np.zeros(config.K[layer])
            thet[index] = 1
            theta.append(thet)
            ind.append(index)
            index = np.argmax(np.array(Phi[layer])[:, index])
        Theta = [theta[2],theta[1],theta[0]]
        # num = 1
        # theta1 = np.zeros([config.K[0]])
        # theta1[topic_id[0]] = num
        # theta2 = np.zeros([config.K[1]])
        # theta2[topic_id[1]] = num
        # theta3 = np.zeros([config.K[2]])
        # theta3[topic_id[2]] = num
        # Theta = [theta1, theta2, theta3]

        return self.generate(sess,config, Theta, start_word_id, temperature, max_length, stop_word_id)

    #generate a sequence of words, given a document
    def generate_on_doc(self, sess, Theta, start_word_id, temperature=1.0, max_length=30, stop_word_id=None):
        return self.generate(sess, Theta, start_word_id, temperature, max_length, stop_word_id)
