# Please run "exp_all.sh" in folder "ocsvm_prob" first.
import os
import sys
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.stats import gamma
from scipy.stats import norm
from matplotlib.collections import PathCollection
from matplotlib.legend_handler import HandlerPathCollection, HandlerLine2D

def rn_gamma(dec_values,gamma_params):
    probs = np.zeros(len(dec_values))
    mean,var,max_dec = gamma_params
    dec = [max_dec - dec_value if dec_value<max_dec else 0 for dec_value in dec_values]

    k = mean*mean/var
    theta = var/mean

    cdf = gamma.cdf(dec,a=k,scale=theta)
    cdf_mu = gamma.cdf(mean,a=k,scale=theta)
    probs = [1-max(0,(c-cdf_mu)/(1-cdf_mu)) for c in cdf]
    return probs

def rn_gaussian(dec_values,gamma_params):
    probs = np.zeros(len(dec_values))
    mean,var,max_dec = gamma_params
    dec = [max_dec - dec_value if dec_value<max_dec else 0 for dec_value in dec_values]

    cdf = norm.cdf(dec,loc=mean,scale=var)
    probs = [1-max(0,2*c-1) for c in cdf]
    return probs

def new_gamma_scaling(dec_values, gamma_params):
    mean, var, max_dec = gamma_params
    dec = [max_dec - dec_value if dec_value<max_dec else 0 for dec_value in dec_values]
    k = mean*mean/var
    theta = var/mean
    cdf = gamma.cdf(dec,a=k,scale=theta)
    probs = 1-cdf
    prob_dec0 = 1-gamma.cdf(max_dec,a=k,scale=theta)
    probs = np.where(probs>=prob_dec0, 0.5+0.5*(probs-prob_dec0)/(1-prob_dec0),0.5*probs/prob_dec0)
    return probs

def plot_relation(fi, gamma_params):
    filename = "../data/output/output_"+fi+".t"
    fileprob = "../data/"+fi+"_prob.t"
    data = []
    with open(filename) as f:
        for line in f:
            data.append([str(n) for n in line.strip().split('\t',5)])

    data_df = pd.DataFrame(data,columns=["label_platt","platt","dec_value","uniform","density","label"])
    dec=[float(i) for i in data_df.dec_value.drop(0).tolist()]
    pla=[float(i) for i in data_df.platt.drop(0).tolist()]
    uni=[float(i) for i in data_df.uniform.drop(0).tolist()]
    den=[float(i) for i in data_df.density.drop(0).tolist()]

    prob = []
    with open(fileprob) as f:
        for line in f:
            prob.append([str(line)])
    prob_df = pd.DataFrame(prob,columns=["probability"])
    pr=[float(i) for i in prob_df.probability.tolist()]

    prob_gamma = new_gamma_scaling(dec, gamma_params)
    prob_rn_gaussian = rn_gaussian(dec, gamma_params)
    prob_rn_gamma = rn_gamma(dec, gamma_params)
    print("MSE of dataset:  "+fi)
    mse_rn_gamma = mean_squared_error(pr,prob_rn_gamma)
    mse_rn_gaussian = mean_squared_error(pr,prob_rn_gaussian)
    print("Gamma RN: "+str(mse_rn_gamma))
    print("Gaussian RN: "+str(mse_rn_gaussian))

    plt.scatter(dec,pr,label="Ideal",s=10.,color='red',alpha=0.1)
    plt.scatter(dec,pla,label="Platt scaling",s=1.,color='coral',alpha=0.1)
    plt.scatter(dec,uni,label="Binning: equidistantly",s=1.,color='green',alpha=0.1)
    plt.scatter(dec,den,label="Binning: by density",s=1.,color='royalblue',alpha=0.1)
    plt.scatter(dec,prob_gamma,label="New Gamma scaling",s=1.,color='black',alpha=0.1)
    plt.scatter(dec,prob_rn_gamma,label="Gamma RN",s=1.,color='goldenrod',alpha=0.1)
    plt.scatter(dec,prob_rn_gaussian,label="Gaussian RN",s=1.,color='lightslategray',alpha=0.1)

    lg=plt.legend(loc='upper left', fontsize='x-large')
    for lh in lg.legendHandles:
        lh.set_alpha(1)
        lh._sizes = [40]
    plt.xlabel("$f$")
    plt.ylabel("$P$(normal|$f$)")
    plt.title(fi)
    plt.show()

if __name__ == "__main__":
    # new Gamma scaling parameters: mean, variance and maximum decision value
    # can be found in data/output/stdout_filename
    # art1, art2, art3(g=0.0001), art3(g=0.1), art_5, art_10, fourclass, usps
    datalist = ['art1','art2','art3','g01art3','art_5','art_10','fourclass_scale_oc','usps_oc']
    gamma_params = [(0.248800,0.122953,0.328571), (0.265098,0.122966,0.335988), (3.039675,0.626432,4.392296), (4.545228,60.535216,13.655819), (1.232163,0.628042,1.645702), (2.223056,1.244507,2.850334), (0.004511,0.000008,0.006402), (0.505399,0.100051,0.654425)]
    
    for i in range(2):
        plot_relation(datalist[i],gamma_params[i])

