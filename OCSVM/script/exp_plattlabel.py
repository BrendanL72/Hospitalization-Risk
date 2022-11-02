# Please run "exp_plattlabel_1.sh" and "exp_plattlabel_2.sh" in folder "ocsvm_prob" before run this.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from matplotlib.collections import PathCollection
from matplotlib.legend_handler import HandlerPathCollection, HandlerLine2D

if __name__ == "__main__":
    prob = []
    with open('../data/art1_prob.t') as f:
        for line in f:
            prob.append([str(line)])
    prob_df = pd.DataFrame(prob,columns=["probability"])
    pr=[float(i) for i in prob_df.probability.tolist()]

    # art1 -g 0.0001 w/o true labels
    data_1 = []
    with open("../data/output_plattlabel/output_wo_g0.0001_art1.t") as f:
        for line in f:
            data_1.append([str(n) for n in line.strip().split('\t',5)])
    data_df = pd.DataFrame(data_1,columns=["label_platt","platt","dec_value","uniform","density","label"])
    dec0001=[float(i) for i in data_df.dec_value.drop(0).tolist()]
    pla0001_wo=[float(i) for i in data_df.platt.drop(0).tolist()]
    
    # art1 -g 0.0001 w/ true labels
    data_2 = []
    with open("../data/output_plattlabel/output_w_g0.0001_art1.t") as f:
        for line in f:
            data_2.append([str(n) for n in line.strip().split('\t',5)])
    data_df = pd.DataFrame(data_2,columns=["label_platt","platt","dec_value","uniform","density","label"])
    pla0001_w=[float(i) for i in data_df.platt.drop(0).tolist()]

    plt.scatter(dec0001,pr,label="Ideal",s=1.,color='red',alpha=0.1)
    plt.scatter(dec0001,pla0001_wo,label=r"$\gamma$=0.0001 w/o true labels",s=1.,color='coral',alpha=0.1)
    plt.scatter(dec0001,pla0001_w,label=r"$\gamma$=0.0001 w/ true labels",s=1.,color='olive',alpha=0.1)
    lg=plt.legend(loc='upper left', fontsize='x-large')
    for lh in lg.legendHandles:
        lh.set_alpha(1)
        lh._sizes = [40]
    plt.xlabel("$f$")
    plt.ylabel("$P$(normal|$f$)")
    plt.show()

    # art1 -g 0.25 w/o true labels
    data_3 = []
    with open("../data/output_plattlabel/output_wo_g0.25_art1.t") as f:
        for line in f:
            data_3.append([str(n) for n in line.strip().split('\t',5)])
    data_df = pd.DataFrame(data_3,columns=["label_platt","platt","dec_value","uniform","density","label"])
    dec25=[float(i) for i in data_df.dec_value.drop(0).tolist()]
    pla25_wo=[float(i) for i in data_df.platt.drop(0).tolist()]
    
    # art1 -g 0.25 w/ true labels
    data_4 = []
    with open("../data/output_plattlabel/output_w_g0.25_art1.t") as f:
        for line in f:
            data_4.append([str(n) for n in line.strip().split('\t',5)])
    data_df = pd.DataFrame(data_4,columns=["label_platt","platt","dec_value","uniform","density","label"])
    pla25_w=[float(i) for i in data_df.platt.drop(0).tolist()]

    plt.scatter(dec25,pr,label="Ideal",s=1.,color='red',alpha=0.1)
    plt.scatter(dec25,pla25_wo,label=r"$\gamma$=0.25 w/o true labels",s=1.,color='violet',alpha=0.1)
    plt.scatter(dec25,pla25_w,label=r"$\gamma$=0.25 w/ true labels",s=1.,color='steelblue',alpha=0.1)
    lg=plt.legend(loc='upper left', fontsize='x-large')
    for lh in lg.legendHandles:
        lh.set_alpha(1)
        lh._sizes = [40]
    plt.xlabel("$f$")
    plt.ylabel("$P$(normal|$f$)")
    plt.xlim(-35,np.max(dec25)+0.3)
    plt.show()
