import ot
import numpy as np
import pandas as pd

def histogram(A, bins):
    """
    :param A:  list of numbers (should be np.array or list type)
    :param bins: score bins, i.e. if scores are all integers 1-100 then the bins should
    be a list that looks like [1,2,...100].
    :return: histogram of scores snapped to the bins as vector

    Assumption: method assumes that bin[i] < bins[i+1] for all i < len(bins)
    also assumes that bin[0] <= all a in A <= bin[-1]

    """
    n = len(bins)
    hist = [0]*n

    for score in A:
        idx = np.abs(score - bins).argmin()
        hist[idx] += 1
    return np.array(hist)


def is_CDF(cdf, ep=1e-4):
    return (cdf[-1] - 1) < ep

def empiricalCDF(sample, bins):
    """
    Computes empirical CDF of sample
    :param sample: a list (of type np.array or list) of numbers
    :param bins: score bins, i.e. if scores are all integers 1-100 then the bins should
    be a list that looks like [1,2,...100].
    :return: returns a np.array X s.t. for each i < len(X), X[i] is the CDF value for
    the score corresponding to bin[i]
    """
    empiricalpdf = empiricalPDF(sample, bins) #get pdf
    proposedCDF = CDF(empiricalpdf) #CDF-ify the pdf

    if is_CDF(proposedCDF):
        proposedCDF = np.around(proposedCDF, 2)
        return proposedCDF
    else:
        exit() #TODO: throw error here

def empiricalPDF(sample, bins):
    """
    Computes empirical PDF of some sample 
    :param sample: a list (of type np.array or list) of numbers
    :param bins: score bins, i.e. if scores are all integers 1-100 then the bins should
    be a list that looks like [1,2,...100].
    :return: returns a np.array X s.t. for each i < len(X), X[i] is the PDF value for
    the score corresponding to bin[i] -- a normalized histogram (if you will)
    """
    hist = histogram(sample, bins=bins) #get histogram
    return hist/np.sum(hist) #normalize histogram

def edm(vect):
    """
    Gets a Euclidean distance matrix for a vector
    :param vect: length n nd.array,
    assumes the vector is bin like, vect[i] < vect[i+1] for all i
    :return: a matrix where each M[i][j] is the distance between i,j
    """

    if len(vect.shape) > 1 and vect.shape[1] == 1:
        vect = vect.flatten()

    return np.abs(vect - np.expand_dims(vect, axis=1))

def CDF(A):
    """
    Turns a PDF into a CDF
    this is just for readability
    :param A: a PDF as an nd.array
    :return: the CDF of A
    """
    return np.cumsum(A, axis=0)

def psuedoInverse(percentile, CDF, bins):
    """
    For some percentile, its pseudo inverse is the score which lives at that percentile
    This is not always well defined, hence why it is a psuedo inverse and not a true inverse
    see "Optimal Transport for Applied Mathematicians" page 54 def 2.1 
    :param percentile: a number between 0,1 
    :param CDF: some valid CDF as an np.array
    :param bins: distrubtion bins, i.e. if possible values for distribution are all integers 1-100
    then the bins should be a list that looks like [1,2,...100].
    :return: the psuedo inverse of the percentile
    """

    CDF = np.insert(np.around(CDF, 5), 0, 0)
    inf = np.subtract(percentile, CDF)
    #See how far percentile is from the CDF percentiles
    #this will help us locate percentile in CDF nearest the percentile that is input
    #to the problem

    inf[ inf <= 0] = np.Infinity
    inf_ix = inf.argmin()

    inverse = bins[inf_ix]

    return inverse

def barycenter(A, weights, bins, reg=1e-3, solver="exact_LP"):
    """

    :param A: list of samples of scores/numbers, should be list or nd.array.
    each list is the samples you wish to compute the barycenter of

    :param weights: the barycenter weights must sum to 1
    :param bins: distrubtion bins, i.e. if possible values for distribution are all integers 1-100
    then the bins should be a list that looks like [1,2,...100].
    :param reg: regularizer parameter if the solver is NOT exact1D
    :param solver: possible inputs are
        - "exact_1D"
        - "bregman"
    :return: returns the empirical barycenter of A[0]...A[n]

    Barycenter is computed using formula  in "Barycenters in the Wasserstein space" section 6.1
    """

    assert np.abs(np.sum(weights) - 1) < 1e-6,  "Sum of weights must add to 1"

    if solver == "anon": # *****
        CDFs = [empiricalCDF(sample,bins) for sample in A] #Get the CDFs for each sample


        bc_pdf_all = []
        for i in range(len(A)):
            bc_i = np.zeros(shape=bins.shape)  # this is where the BC is stored
            mi_pdf = empiricalPDF(A[i], bins) #the source measure used for the pushforward
            for x in bins:
                bin_ix = np.abs(bins-x).argmin() #find ix of closest bin for some score
                Q_F_m0 = np.array([psuedoInverse(CDFs[i][bin_ix],cdf,bins) for cdf in CDFs])
                #compute psuedo inverse across all of the input sample
                new_score = np.dot(Q_F_m0, weights) #compute new score using BC formula

                newscore_bin_ix = np.abs(bins-new_score).argmin()
                # find ix of the bin of the new score

                bc_i[newscore_bin_ix] += mi_pdf[bin_ix]
                #give the new score the probability of the source measure
            bc_pdf_all.append(bc_i)

        bc_pdf = np.array(bc_pdf_all).mean(axis=0)
    elif solver == "exact_LP":
        weights = np.array(weights)
        A = np.vstack([empiricalPDF(a, bins) for a in A]).T

        M = edm(bins)
        M = np.divide(M,M.max())
        bc_pdf = ot.lp.barycenter(A, M, weights, solver='interior-point', verbose=True)
    elif solver == "bregman":
        weights = np.array(weights)
        A = np.vstack([empiricalPDF(a, bins) for a in A]).T
        M = edm(bins)
        M = np.divide(M,M.max())
        bc_pdf = ot.bregman.barycenter(A, M, reg, weights)
    else:
        assert "not valid solver type"
    return bc_pdf

def transport(x, src, dst, bins):
    """

    :param x: the score to be transported, some val s.t. bin[0] <= x <= bin[-1]
    :param src: the source distribution in the transport, given as a list of values
    :param dst: the destination distribution, assumed to be a barycenter and given as a pdf
    :param bins: distrubtion bins, i.e. if possible values for distribution are all integers 1-100
    then the bins should be a list that looks like [1,2,...100].
    :return: the transported value
    """


    src_cdf = empiricalCDF(src,bins)
    dst_cdf = CDF(dst)
    bin_ix = np.abs(bins - x).argmin()
    q = src_cdf[bin_ix]
    transported_score = psuedoInverse(q,dst_cdf,bins=bins)

    return transported_score


def geometric_adjustment(train_df, test_df, sens_col, score_col, solver, bins, return_barycenter=False):
    """

    :param train_df: training df
    :param test_df: testing df
    :param sens_col: column name of sensitive attribute
    :param score_col: column name containing the scores
    :param bins: distrubtion bins, i.e. if possible values for distribution are all integers 1-100
    then the bins should be a list that looks like [1,2,...100]
    :param solver: the barycenter solver to be used
    :param return_barycenter: (bool)
    :return: returns a data frame with an "adjust" column and "repaired score" column
    """
    groups = [group[1] for group in train_df.groupby(by=sens_col)] #get dataframe groups
    samples = np.array([g[score_col].to_numpy() for g in groups]) #get the scores for each group

    freq = np.array([len(group) for group in groups])
    weights = freq/np.sum(freq)

    bc = barycenter(A=samples,
                   weights=weights,
                   bins=bins,
                   reg=1e-3,
                   solver=solver)

    # test_groups = [group[1] for group in test_df.groupby(by=sens_col)]
    # repaired_dfs = []
    # for i in range(len(test_groups)):
    #     test_group_df = test_groups[i].copy()
    #     test_group_df["repaired_score"] = test_group_df[score_col].apply(
    #         func=transport,
    #         args=(test_group_df[score_col].to_numpy(), bc, bins)
    #     )
    #     test_group_df["shift"] = test_group_df["repaired_score"] - test_group_df[score_col]
    #     repaired_dfs.append(test_group_df)

    # new_repaired_df = pd.concat(repaired_dfs).copy()

    # **** mods to preserve indices
    df_repair = test_df.copy()
    for group in test_df[sens_col].unique():
        df_repair.loc[df_repair[sens_col] == group, "repaired_score"] = df_repair.loc[df_repair[sens_col] == group, score_col].apply(
                        func=transport,
                        args=(df_repair.loc[df_repair[sens_col] == group, score_col].to_numpy(), bc, bins)
                    )

    df_repair["adjust"] = df_repair["repaired_score"] - df_repair[score_col]

    if not return_barycenter:
        return df_repair
    else:
        return df_repair, bc





