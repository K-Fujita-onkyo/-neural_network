from cmath import exp
import numpy as np

def softmax(a):

    c=np.max(a)
    exp_a=np.exp(a-c)#Countermeasure for Override
    sum_exp_a=np.sum(exp_a)
    y=exp_a/sum_exp_a

    return y

if __name__ == '__main__':
    softmax()
