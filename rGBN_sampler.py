
import threading
import GBN_sampler
import numpy as np
import config as cf

class MyThread(threading.Thread):
    def __init__(self, count,X_train,Phi, Theta1, Theta2, Theta3, L, K, T, Pi):
        super(MyThread, self).__init__()
        self.count = count
        self.X_train = X_train
        self.Phi =  Phi


        self.L = L
        self.K = K
        self.T = T
        self.Pi = Pi

        self.Theta = [0]*self.L

        self.Theta[0], self.Theta[1],self.Theta[2] =  Theta1, Theta2, Theta3

    def run(self):

            L_dotkt = [0] *  self.L
            L_kdott = [0] *  self.L
            A_KT = [0] *  self.L
            self.A_VK = [0] *  self.L
            self.L_KK = [0] *  self.L

            prob1 = [0] *  self.L
            prob2 = [0] *  self.L
            Xt_to_t1 = [0] * ( self.L + 1)
            X_layer  = [0] *  self.L
            X_layer_split1 = [0] * self.L

            self.X_train = np.array(self.X_train, order='C').astype('double')


            for l in range(self.L):
                self.L_KK[l] = np.zeros(( self.K[l],  self.K[l]))
                L_dotkt[l] = np.zeros((self.K[l], self.T + 1))
                L_kdott[l] = np.zeros((self.K[l], self.T + 1))

                self.Phi[l] = np.array(self.Phi[l], order='C').astype('double')
                self.Theta[l] = np.array(self.Theta[l], order='C').astype('double')
                self.Pi[l] = np.array(self.Pi[l], order='C').astype('double')
                if l == 0:
                    [A_KT[l],  self.A_VK[l]] = GBN_sampler.Multrnd_Matrix(self.X_train, self.Phi[l], self.Theta[l])
                else:
                    [A_KT[l], self.A_VK[l]] = GBN_sampler.Multrnd_Matrix(Xt_to_t1[l],  self.Phi[l],  self.Theta[l])

                if l ==  self.L - 1:
                    for t in range( self.T - 1, 0, -1):  # T-1 : 1
                        tmp1 = A_KT[l][:, t:t + 1] + L_dotkt[l][:, t + 1:t + 2]
                        tmp2 = cf.tao0 * np.dot(self.Pi[l], self.Theta[l][:, t - 1:t])
                        L_kdott[l][:, t:t + 1] = GBN_sampler.Crt_Matrix(np.array(tmp1, order='C').astype('double'),
                                                                         np.array(tmp2, order='C').astype('double'))

                        [L_dotkt[l][:, t:t + 1], tmp] = GBN_sampler.Multrnd_Matrix(np.array(L_kdott[l][:, t:t + 1], dtype=np.double, order='C'),  self.Pi[l],
                                                         np.array( self.Theta[l][:, t - 1:t], dtype=np.double, order='C'))
                        self.L_KK[l] =  self.L_KK[l] + tmp
                else:
                    prob1[l] = cf.tao0 * np.dot( self.Pi[l],  self.Theta[l])
                    prob2[l] = cf.tao0 * np.dot( self.Phi[l + 1],  self.Theta[l + 1])
                    X_layer[l] = np.zeros(( self.K[l],  self.T, 2))
                    Xt_to_t1[l + 1] = np.zeros(( self.K[l],  self.T))
                    X_layer_split1[l] = np.zeros(( self.K[l], self. T))

                    for t in range(self.T - 1, 0, -1):

                        L_kdott[l][:, t:t + 1] = GBN_sampler.Crt_Matrix(np.array(A_KT[l][:, t:t + 1] + L_dotkt[l][:, t + 1:t + 2], order='C').astype('double'),
                                                                         np.array(cf.tao0 * (np.dot( self.Phi[l + 1], self.Theta[l + 1][:,t:t + 1]) + np.dot(self.Pi[l],  self.Theta[l][:, t - 1:t] )), order='C').astype('double'))
                        tmp_input = np.array(L_kdott[l][:, t:t + 1], dtype=np.float64, order='C')
                        tmp1 = np.array(prob1[l][:, t - 1:t], dtype=np.float64, order='C')
                        tmp2 = np.array(prob2[l][:, t:t + 1], dtype=np.float64, order='C')
                        [tmp, X_layer[l][:, t, :]] = GBN_sampler.Multrnd_Matrix(tmp_input,
                                                                                 np.concatenate((tmp1, tmp2), axis=1),
                                                                                 np.ones((2, 1)))
                        X_layer_split1[l][:, t] = np.reshape(X_layer[l][:, t, 0], -1)  
                        Xt_to_t1[l + 1][:, t] = np.reshape(X_layer[l][:, t, 1], -1)  

                        tmp_input = np.array(X_layer_split1[l][:, t:t + 1], dtype=np.float64, order='C')
                        [L_dotkt[l][:, t:t + 1], tmp] = GBN_sampler.Multrnd_Matrix(tmp_input,  self.Pi[l],
                                                                                    np.array( self.Theta[l][:, t - 1:t],
                                                                                             order='C'))
                        self.L_KK[l] =  self.L_KK[l] + tmp

                    L_kdott[l][:, 0:1] = GBN_sampler.Crt_Matrix(A_KT[l][:, 0:1] + L_dotkt[l][:, 1:2],
                                                                 cf.tao0 * np.dot( self.Phi[l + 1],
                                                                                          self.Theta[l + 1][:, 0:1]))
                    Xt_to_t1[l + 1][:, 0:1] = L_kdott[l][:, 0:1]




    def get_result(self):
        '''
        return
        '''
        return self.count, self.A_VK,  self.L_KK
