# Please run "exp_all.sh" in file "ocsvm_prob" before run this.
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma, percentileofscore
import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqplot_2samples

def get_quantiles(dec_values):
    q = np.array([percentileofscore(dec_values,dec_value)/100 for dec_value in dec_values])
    return q

def get_zero_quantile(dec_values,q):
    idx = np.abs(dec_values).argmin()
    return q[idx]

def rescaled_quantile(dec_values):
    # use dec_values of training data to get the groung truth quantile
    q = get_quantiles(dec_values)
    zq = get_zero_quantile(dec_values,q) # quantile of 0 dec_value (or the nearest)
    for i in range(len(q)):
        if q[i]>=zq:
            q[i] = (q[i]-zq)/(1-zq)/2+0.5
        else:
            q[i] = q[i]/zq*0.5
    return q

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

def qqplot(fi, gamma_params):
    filename = "../data/output/output_"+fi+".t"
    filetrain = "../data/output/output_"+fi
    data = []
    with open(filename) as f:
        for line in f:
            data.append([str(n) for n in line.strip().split('\t',5)])

    data_df = pd.DataFrame(data,columns=["label_platt","platt","dec_value","uniform","density","label"])
    dec=[float(i) for i in data_df.dec_value.drop(0).tolist()]
    pla=[float(i) for i in data_df.platt.drop(0).tolist()]
    uni=[float(i) for i in data_df.uniform.drop(0).tolist()]
    den=[float(i) for i in data_df.density.drop(0).tolist()]

    data_train = []
    with open(filetrain) as f:
        for line in f:
            data_train.append([str(n) for n in line.strip().split('\t',5)])

    data_train_df = pd.DataFrame(data_train,columns=["label_platt","platt","dec_value","uniform","density","label"])
    dec_train=[float(i) for i in data_train_df.dec_value.drop(0).tolist()]

    prob_gamma = new_gamma_scaling(dec, gamma_params)
    ground = rescaled_quantile(dec_train)

    pp_sample = sm.ProbPlot(np.array(pla))
    pp_theo = sm.ProbPlot(np.array(ground))
    qqplot_2samples(pp_sample, pp_theo, ylabel='Sample Quantiles', xlabel='Theoretical Quantiles', line='45')
    plt.title(fi+"  Platt scaling")
    plt.show()

    pp_sample = sm.ProbPlot(np.array(den))
    pp_theo = sm.ProbPlot(np.array(ground))
    qqplot_2samples(pp_sample, pp_theo, ylabel='Sample Quantiles', xlabel='Theoretical Quantiles', line='45')
    plt.title(fi+"  Binning by density")
    plt.show()

    pp_sample = sm.ProbPlot(np.array(prob_gamma))
    pp_theo = sm.ProbPlot(np.array(ground))
    qqplot_2samples(pp_sample, pp_theo, ylabel='Sample Quantiles', xlabel='Theoretical Quantiles', line='45')
    plt.title(fi+"  New Gamma scaling")
    plt.show()

if __name__ == "__main__":
    # new Gamma scaling parameters: mean, variance and maximum decision value
    # can be found in data/output/stdout_filename
    # art1, art2, art3(g=0.0001), art3(g=0.1), art_5, art_10, fourclass, usps
    datalist = ['art1','art2','art3','g01art3','art_5','art_10','fourclass_scale_oc','usps_oc']
    gamma_params = [(0.248800,0.122953,0.328571), (0.265098,0.122966,0.335988), (3.039675,0.626432,4.392296), (4.545228,60.535216,13.655819), (1.232163,0.628042,1.645702), (2.223056,1.244507,2.850334), (0.004511,0.000008,0.006402), (0.505399,0.100051,0.654425)]

    for i in [0,1,3,6,7]:
        qqplot(datalist[i],gamma_params[i])

