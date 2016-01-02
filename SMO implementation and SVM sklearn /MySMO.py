# This is simplified SMO class
# It can predict for rbf and linear kernels
__author__ = 'vaibhavtyagi'
import numpy as np
from numpy import *

class MySMO():


    # Code for RBF kernel
    def rbf(x, y, sigma=2.0):
        gamma = 1 / (2 * (sigma ** 2))
        if x.shape[0] == 1 and y.shape[0] == 1:
            squared_euclidian = np.sum(np.square(np.subtract(y, x)))
        else:
            squared_euclidian = np.sum(np.square(np.subtract(y, x)), axis=1)
        return np.exp(-gamma*squared_euclidian)

    # code for linear kernel
    def dot_product(self,x, y):
        return np.dot(np.asarray(x)[0], np.asarray(y)[0]) if x.shape[0] == 1 and y.shape[0] == 1 else np.dot(x, y)


    def select_jrand(self,i, m):
        j = i
        while (j == i):
            j= int(random.uniform(0, m))
        return j

    def clip_alpha(self,aj,H,L):
        if aj > H:
            aj = H
        if L > aj:
            aj = L
        return aj

    # Simplified SMO algorithm implementation 
    def smo_simple(self,dataMatIn, class_labels, C, toler, maxIter):
        gram_matrix = np.empty((class_labels.size, class_labels.size), dtype=np.float32)
        for i in xrange(dataMatIn.shape[0]):
            gram_matrix[i, :] = self.dot_product(dataMatIn, dataMatIn[i])
        gram_matrix = mat(gram_matrix)
        #print gram_matrix
        dataMatrix = mat(dataMatIn)
        label_mat = mat(class_labels).transpose()

        b = 0
        m, n = shape(dataMatrix)
        alphas = mat(zeros((m, 1)))
        upper_iter = 0
        iter = 0
        while iter < maxIter:
            print 
            print
            print "***********"
            print "Iteration",upper_iter
            upper_iter += 1
            if upper_iter == 1501:
                break

            alpha_pairs_changed = 0
            for i in range(m):
                fXi = float(multiply(alphas, label_mat).T * gram_matrix[:, i]) + b
                Ei = fXi - float(label_mat[i])
                #print Ei

                if ((label_mat[i]*Ei < -toler) and (alphas[i] < C)) or ((label_mat[i]*Ei > toler) and (alphas[i] > 0)):
                    j = self.select_jrand(i,m)
                    fXj = float(multiply(alphas, label_mat).T * gram_matrix[:, j]) + b

                    Ej = fXj - float(label_mat[j])

                    alpha_i_old = alphas[i].copy()
                    alpha_j_old = alphas[j].copy()
                    if (label_mat[i] != label_mat[j]):
                        L = max(0, alphas[j] - alphas[i])
                        H = min(C, C + alphas[j] - alphas[i])
                    else:
                        L = max(0, alphas[j] + alphas[i] - C)
                        H = min(C, alphas[j] + alphas[i])

                    if L == H:
                        continue

                    eta = 2.0 * np.atleast_2d(gram_matrix[i, j] - gram_matrix[i, i] - gram_matrix[j, j])

                    if eta >= 0:
                        continue

                    alphas[j] -= label_mat[j] * (Ei - Ej) / eta
                    alphas[j] = self.clip_alpha(alphas[j], H, L)

                    if abs(alphas[j] - alpha_j_old) < 0.00001:
                        continue

                    alphas[i] += label_mat[j] * label_mat[i] * (alpha_j_old - alphas[j])

                    b1 = b - Ei - label_mat[i] * (alphas[i] - alpha_i_old) * np.atleast_2d(gram_matrix[i, i]) - label_mat[j] * (alphas[j] - alpha_j_old) * np.atleast_2d(gram_matrix[i, j])
                    b2 = b - Ej - label_mat[i] * (alphas[i] - alpha_i_old) * np.atleast_2d(gram_matrix[i, j]) - label_mat[j] * (alphas[j] - alpha_j_old) * np.atleast_2d(gram_matrix[j, j])

                    if (0 < alphas[i]) and (C > alphas[i]):
                        b = b1
                    elif (0 < alphas[j]) and (C > alphas[j]):
                        b = b2
                    else:
                        b = (b1 + b2) / 2.0
                    alpha_pairs_changed += 1

            print 'No of pairs changed',alpha_pairs_changed
            if alpha_pairs_changed == 0:
                iter += 1
            else:
                iter = 0

        return b, alphas
