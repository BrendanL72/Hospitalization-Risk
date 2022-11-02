# Please run "i_ocsvm.py" in folder "ocsvm_prob" before run this.
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error

def get_decval(filename):
    data = []
    with open(filename) as f:
            for line in f:
                    data.append([str(n) for n in line.strip().split('\t',5)])
    data_df = pd.DataFrame(data,columns=["label_platt","platt","dec_value","uniform","density","label"])
    dec=[float(i) for i in data_df.dec_value.drop(0).tolist()]
    return dec


def mse(fi,nu):
    filename = fi
    filepath = "../data/output_i_ocsvm_"+fi+"/"
    if fi == 'art3_g_0.1' or fi == 'art3_g_0.0001':
        filename = 'art3'
    index = np.zeros(11) # data index of instances on boundary of \nu=...
    # \nu : 0.05, 0.10, 0.15, 0.20, 0.25
    for i in range(1,10):
        ii=nu[i-1]
        #print(i)
        filen = filepath+"output_"+filename+"_"+format(ii,'.2f')
        dec = get_decval(filen)
        for j in range(10000):
                dec[j]=abs(dec[j])
        index[i]=dec.index(min(dec))
        #print(dec[index[i].astype(int)])

    dec = get_decval(filepath+"output_"+filename+"_"+format(nu[4],'.2f'))
    #print(format(nu[4],'.2f'))

    nu_val = np.zeros(11)
    for i in range(1,10):
        nu_val[i] = dec[index[i].astype(int)]
    nu_val[0] = np.min(dec)
    nu_val[10] = np.max(dec)

    marks = np.zeros(10)
    for i in range(10):
        marks[i] = (nu_val[i]+nu_val[i+1])/2
        #print(marks[i])

    dec_test = get_decval(filepath+"output_"+filename+"_"+format(nu[4],'.2f')+".t")
    ioc_prob = np.zeros(10000)
    for i in range(10000):
        #print(dec_test[i])
        ioc_prob[i] = 0.999
        for j in range(10):
            if dec_test[i]<marks[j]:
                ioc_prob[i] = format(j*0.1,'.1f')
                break
        if ioc_prob[i]==0:
            ioc_prob[i]=0.001

    prob = []
    with open("../data/"+filename+"_prob.t") as f:
        for line in f:
            prob.append([str(line)])
    prob_df = pd.DataFrame(prob,columns=["probability"])
    pr=[float(i) for i in prob_df.probability.tolist()]
    mse_ioc = mean_squared_error(pr,ioc_prob)
    print(mse_ioc)


if __name__ == "__main__":
    for filename in ['art1','art2','art_5','art_10']:
        print("MSE of dataset: "+filename)
        mse(filename,[0.05,0.10,0.15,0.20,0.25,0.40,0.55,0.70,0.85])
    
    for filename in ['art3_g_0.1','art3_g_0.0001']:
        print("MSE of dataset :"+filename)
        mse(filename,[0.01,0.02,0.03,0.04,0.05,0.24,0.43,0.62,0.81])
