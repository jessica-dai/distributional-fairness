
import src.data_2 as data
from src.bcmap import *
import ot
import seaborn as sns
from src.lexi import lexicographicOptimizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def uniformBarycenterTest1():
    k = 100
    A = np.random.uniform(0,1,k)
    B = np.random.uniform(0,1,k)
    bins = np.linspace(0,1,101)
    p = .5
    bc = CDF(barycenter([A,B],[p,1-p], bins=bins))

    plt.plot(empiricalCDF(A,bins))
    plt.plot(empiricalCDF(B,bins))
    plt.plot(bc)
    plt.show()

def uniformBarycenterTest2():
    k = 100
    A = np.random.uniform(0,1,k)
    A = np.minimum(A + .2,1)
    B = np.random.uniform(0,1,k)
    bins = np.linspace(0,1,101)
    p = .5
    bc = CDF(barycenter([A,B],[p,1-p], bins=bins))

    plt.plot(empiricalCDF(A,bins))
    plt.plot(empiricalCDF(B,bins))
    plt.plot(bc)
    plt.show()



def testGaussianBarycenter1():
    n= 100
    k= int(1e3)
    a1 = ot.datasets.make_1D_gauss(n, m=40, s=10)  # m= mean, s= std
    a2 = ot.datasets.make_1D_gauss(n, m=60, s=8)


    a1_l = np.random.choice(np.linspace(0,1,n), k, p=a1)
    a2_l = np.random.choice(np.linspace(0,1,n), k, p=a2)

    bins = np.linspace(0,1,101)
    p = .5
    bc = CDF(barycenter([a1_l,a2_l],[p,1-p], bins=bins))

    plt.plot(empiricalCDF(a1_l,bins))
    plt.plot(empiricalCDF(a2_l,bins))
    plt.plot(bc)
    plt.show()

def testBarycenterofThree():
    n= 100
    k= int(1e3)

    a1 = ot.datasets.make_1D_gauss(n, m=40, s=10)  # m= mean, s= std
    a2 = ot.datasets.make_1D_gauss(n, m=60, s=8)
    a3 = ot.datasets.make_1D_gauss(n, m=15, s=2)



    a1_l = np.random.choice(np.linspace(0,1,n), k, p=a1)
    a2_l = np.random.choice(np.linspace(0,1,n), k, p=a2)
    a3_l = np.random.choice(np.linspace(0,1,n), k, p=a3)


    bins = np.linspace(0,1,101)
    bc = CDF(barycenter([a1_l,a2_l, a3_l],[.25,.5,.25], bins=bins))

    plt.plot(empiricalCDF(a1_l,bins))
    plt.plot(empiricalCDF(a2_l,bins))
    plt.plot(empiricalCDF(a3_l,bins))

    plt.plot(bc, label="bc")
    plt.legend()
    plt.show()

def testCompas():

    B = "African-American"
    W = "Caucasian"
    bins = np.arange(1, 11)

    df = data.compas().sample(frac=1)
    n = len(df)
    alpha = .8
    split_ix = int(n * alpha)

    df_train = df.iloc[:split_ix]

    black_train = df_train[df_train[B] == True]
    white_train = df_train[df_train[W] == True]

    df_train = pd.concat([black_train, white_train])

    p = len(black_train)/len(df_train)
    A = [black_train["decile_score"].to_numpy(),white_train["decile_score"].to_numpy()]
    bc = CDF(barycenter(A,[p,1-p], bins=bins))

    plt.plot(empiricalCDF(black_train["decile_score"], bins), label="black")
    plt.plot(empiricalCDF(white_train["decile_score"], bins), label="white")
    plt.plot(bc, label="Barycenter")
    plt.legend()

    plt.show()

def testInverse():
    n = 100
    k = int(1e3)

    a1 = ot.datasets.make_1D_gauss(n, m=40, s=10)  # m= mean, s= std
    a2 = ot.datasets.make_1D_gauss(n, m=60, s=8)


    bins = np.linspace(0,1, 101)

    inv1 = [psuedoInverse(q, CDF(a1), bins) for q in bins]
    inv2 = [psuedoInverse(q, CDF(a2), bins) for q in bins]

    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(CDF(a1))
    ax1.plot(CDF(a2))

    ax2.plot(inv1)
    ax2.plot(inv2)
    plt.show()



def demoGeometricRepairCompas():
    B = "African-American"
    W = "Caucasian"
    bins = np.arange(1, 11)

    df = data.compas().sample(frac=1)
    n = len(df)
    alpha = .8
    split_ix = int(n * alpha)

    df_train = df.iloc[:split_ix]
    df_test = df.iloc[split_ix:]


    black_train = df_train[df_train[B] == True]
    white_train = df_train[df_train[W] == True]

    df_train = pd.concat([black_train, white_train])

    repaired, bc = geometric_adjustment(train_df=df_train,
                         test_df=df_test,
                         sens_col=B,
                         score_col="decile_score",
                         solver="exact_LP",
                         bins=bins,
                         return_barycenter=True)

    f, axs = plt.subplots(2, 3, figsize=(8,6))
    axs[0,0].hist(df_test[df_test[B] == True]["decile_score"].to_numpy(), density=True, label="Black(Unrepaired)")
    axs[0,1].hist(repaired[repaired[B] == True]["repaired_score"].to_numpy(), density=True, label="Black(Repaired)")

    mapping = [transport(x=x,
                         src=df_test[df_test[B] == True]["decile_score"].to_numpy(),
                         dst=bc, bins=bins)
               for x in bins]
    axs[0,2].scatter(x=bins, y=mapping, label="black T")

    axs[1, 0].hist(df_test[df_test[W] == True]["decile_score"].to_numpy(), density=True, label="White(Unrepaired)")
    axs[1, 1].hist(repaired[repaired[W] == True]["repaired_score"].to_numpy(), density=True, label="White(Repaired)")

    mapping = [transport(x=x,
                         src=df_test[df_test[W] == True]["decile_score"].to_numpy(),
                         dst=bc, bins=bins)
               for x in bins]
    axs[1, 2].scatter(x=bins, y=mapping, label="white T")
    _ = [ax.legend() for ax in axs.flatten()]
    plt.show()

def testFicoLex():
    n = 1000

    fico_df = data.FicoDataset(n=n)

    bins = np.arange(0, 100.5, .5)

    #Compute the barycenter of the distributions with the perscribed weights
    repaired, bc = geometric_adjustment(train_df=fico_df,
                                        test_df=fico_df,
                                        sens_col="group",
                                        score_col="score",
                                        solver="bregman",
                                        bins=bins,
                                        return_barycenter=True)


    data_pos_label = repaired[repaired["label"] == 1].copy()

    #repaired columns are called "lexi" and "maxin"
    lex_df = lexicographicOptimizer(
        df=data_pos_label,
        attr_col="group",
        score_col="score",
        shift_col="shift"
    )




    f, axs = plt.subplots(3, 4, figsize=(8,6))
    groups = [gb for gb in lex_df.groupby("group")]

    for i in range(len(groups)):
        uh, group = groups[i]
        axs[0,i].hist(group["score"], density=True, label=uh)
        axs[1,i].hist(group["maxmin"], density=True, label=uh)
        axs[2,i].hist(group["lexi"], density=True, label=uh)

    plt.show()






testFicoLex()