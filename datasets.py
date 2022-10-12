import numpy as np
import pandas as pd 

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, normalize
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from folktables import adult_filter, public_coverage_filter, ACSIncome, ACSPublicCoverage, ACSMobility, ACSEmployment, BasicProblem
from aif360.datasets.adult_dataset import AdultDataset
from aif360.datasets.binary_label_dataset import BinaryLabelDataset

def get_data(dataset, seed, verb, algo="rf"):
    """
    this is used in the main `run.py` script (i.e. for our own algorithm)
    fair baselines use the datasets directly

    `algo` can be rf, lr, svm, mlp
    """
    if dataset == 'taiwan':
        return gen_taiwan_credit(seed=seed, verb=verb, algo=algo)
    elif dataset == 'german':
        return gen_german_probs(seed=seed, verb=verb)
    elif dataset == 'adult_old':
        return gen_adult_probs(seed=seed, verb=verb, algo=algo)
    elif dataset == 'adult_new':
        return gen_new_adult(seed=seed, verb=verb, task='income', algo=algo)
    elif dataset == 'public':
        return gen_new_adult(seed=seed, verb=verb, task='public', algo=algo)
    elif dataset == 'employment':
        return gen_new_adult(seed=seed, verb=verb, task='employment_sex', algo=algo)

    else: 
        print("dataset typo?")

def _get_model(modelname, seed=0):
    if modelname == 'rf':
        print("returning a rf")
        return RandomForestClassifier(random_state=seed)
    elif modelname == 'lr':
        print("returning a lr")
        return LogisticRegression(random_state=seed)
    elif modelname == 'mlp':
        print("returning a mlp")
        return MLPClassifier(random_state=seed)
    elif modelname == 'svm':
        print("returning a svm")
        return SVC(random_state=seed, probability=True)

def gen_taiwan_credit(seed=0, verb=False, interv=False, algo="rf"):
    """
    """
    credit = pd.read_csv('data_raw/UCI_Credit_Card.csv') # 30k rows

    credit['EDUCATION'] = credit['EDUCATION'] < 2
    labels = credit['default.payment.next.month'].values
    sens = credit['EDUCATION'].values
    X = credit.drop(columns=['default.payment.next.month']) #, 'EDUCATION']) 

    if interv:
        bld = BinaryLabelDataset(df=credit, label_names=['default.payment.next.month'], protected_attribute_names=['EDUCATION'])
        return bld.split(1, shuffle=True, seed=seed)       

    X_train, X_test, y_train, y_test, group_train, group_test = train_test_split(
        X, labels, sens, test_size=0.6667, random_state=seed)

    classifier = _get_model(algo, seed=seed)
    model = make_pipeline(StandardScaler(), classifier)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)

    ret = pd.DataFrame(columns=['score', 'group', 'label'])
    ret['score'] = probs[:,1]
    ret['label'] = y_test
    ret['group'] = group_test

    if verb:
        print("trained on:", len(X_train), "samples")
        print("returning: ", len(ret), "samples")

    return ret

def gen_german_probs(seed=0, verb=False, sens='age'):
    """
    note: did drop sens in jan 22 preprocessing
    """
    german = pd.read_csv('data_raw/german.csv') # from friedler 2019 preprocessing
    german["age"] = np.array([1 if a == 'adult' else 0 for a in german.age]).astype(int)
    german['sex'] = np.array([1 if a == 'male' else 0 for a in german.sex]).astype(int)
    german['credit'] = np.array([1 if a == 2 else 0 for a in german.credit]).astype(int)
    sens = german[sens]
    labels = german['credit'].astype(int)
    #Get rid of the columns we don't want
    X = german.drop(columns=['sex-age', 'credit','age', 'sex'])
    X = X.values
    good_inds = ~np.isnan(X).any(axis=1) # all inds are good
    X = X[good_inds]
    labels = labels[good_inds].values
    sens = sens[good_inds].values
    # scale and shuffle
    ss = MinMaxScaler().fit(X)
    X = normalize(ss.transform(X))
    shuffleinds = np.arange(len(X))
    np.random.seed(seed)
    np.random.shuffle(shuffleinds)
    X = X[shuffleinds]
    labels = labels[shuffleinds]
    sens = sens[shuffleinds]
    # train and predict
    split_ind = int(len(X)/2)
    classifier = RandomForestClassifier(random_state=seed).fit(X[:split_ind], labels[:split_ind])
    probs = classifier.predict_proba(X[split_ind:])
    ret = pd.DataFrame(columns=['score', 'group', 'label'])
    ret['score'] = probs[:,1]
    ret['label'] = labels[split_ind:].astype(np.int)
    ret['group'] = sens[split_ind:]

    if verb:
        print("trained on:", split_ind, "samples")
        print("returning: ", len(ret), "samples")
    return ret

def gen_adult_probs(seed=0,verb=False, sens='sex', interv=None, algo='rf'):
    """
    Old adult dataset
    New for experiments may 22 -- ibm implementation feldman method requires using their dataset formatting
    """
    adult = AdultDataset(protected_attribute_names=[sens],
                         privileged_classes=[['Male']], 
                         categorical_features=[],
                         features_to_keep=['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week'])

    # scale
    scaler = MinMaxScaler(copy=False)
    adult.features = scaler.fit_transform(adult.features)

    # return just the scaled and shuffled data if pre/inprocessing
    if interv == 'pre/in':
        return adult.split(1, shuffle=True, seed=seed)[0] # will eventually be split into 3

    # otherwise split the data and train a classifier (don't drop sensitive feature)
    # sensind = adult.feature_names.index('sex')
    # adult.features = np.delete(adult.features, sensind, axis=1)
    train, test = adult.split([0.3333333], shuffle=True, seed=seed) # the "test" portion will eventually be split again

    classifier = _get_model(algo, seed=seed)
    classifier.fit(train.features, train.labels.reshape(-1))
    probs = classifier.predict_proba(test.features)

    ret = pd.DataFrame(columns=['score', 'group', 'label'])
    ret['score'] = probs[:,1]
    ret['label'] = test.labels
    ret['group'] = test.protected_attributes

    if verb:
        print("trained on:", len(train.labels), "samples")
        print("returning: ", len(ret), "samples")

    return ret

def gen_adult_probs_oldpreprocess(seed=0, verb=False, sens='sex', interv=None):
    """
    note: did not drop sens in jan 22 experiments

    this method used for experiments pre may 22 
    """
    adult = pd.read_csv('data_raw/adult_numerical.csv') # from friedler 2019 preprocessing
    # dataset wrangling - total 30162 
    X = adult.drop(columns=['income-per-year', 'race-sex'])
    X['race'] = np.array([1 if a == 'White' else 0 for a in X.race])
    X['sex'] = np.array([1 if a == 'Male' else 0 for a in X.sex])
    sensv = X[sens]
    X = X.drop(columns=[sens]).values # new apr 22
    labels = np.ones(len(X))
    labels[adult['income-per-year'] == '<=50K'] = 0

    good_inds = ~np.isnan(X).any(axis=1) # all inds are good
    X = X[good_inds]
    labels = labels[good_inds]
    sensv = sensv[good_inds].values

    # scale and shuffle
    ss = MinMaxScaler().fit(X)
    X = normalize(ss.transform(X))

    np.random.seed(seed)
    shuffleinds = np.arange(len(X))
    np.random.shuffle(shuffleinds)

    X = X[shuffleinds]
    labels = labels[shuffleinds]
    sensv = sensv[shuffleinds]

    if interv == 'pre/in':
        return X, labels, sensv

    # train and predict
    split_ind = 15000
    classifier = RandomForestClassifier(random_state=seed).fit(X[:split_ind], labels[:split_ind])
    probs = classifier.predict_proba(X[split_ind:])

    ret = pd.DataFrame(columns=['score', 'group', 'label'])
    ret['score'] = probs[:,1]
    ret['label'] = labels[split_ind:]
    ret['group'] = sensv[split_ind:]

    if verb:
        print("trained on:", split_ind, "samples")
        print("returning: ", len(ret), "samples")

    if interv == 'post':
        return ret, X[split_ind:]

    return ret

def _get_employment_sex():
    return BasicProblem(features=ACSEmployment.features, 
                                 target=ACSEmployment.target,
                                 group='SEX',
                                 group_transform=lambda x: x > 1,
                                 target_transform=ACSEmployment.target_transform,
                                 postprocess=lambda x: np.nan_to_num(x, -1))

def gen_new_adult(seed, verb, task='income', interv=None, algo='rf'):
    """
    """

    tasks = {
        'income': ACSIncome,
        'public': ACSPublicCoverage,
        'mobility': ACSMobility,
        'employment_sex': _get_employment_sex()
    }
    # data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    # if verb: 
    #     print("downloading data")
    acs_data = pd.read_csv('../data_raw/acs_data.csv') # data_source.get_data(states=['CA']) #, download=True)

    if verb:
        print("data downloaded")

    if interv is not None:
        if task == 'income': # this does the work in `df_to_numpy` below
            filtered_df = adult_filter(acs_data)[ACSIncome.features + [ACSIncome.target]] 
            filtered_df[ACSIncome.target] = ACSIncome.target_transform(filtered_df[ACSIncome.target]).astype(int)
            filtered_df[ACSIncome.group] = [1 if gp == 1 else 0 for gp in filtered_df[ACSIncome.group]]
        elif task == 'public':
            filtered_df = public_coverage_filter(acs_data)[ACSPublicCoverage.features].apply(lambda x: np.nan_to_num(x, -1))
            filtered_df[ACSPublicCoverage.target] = ACSPublicCoverage.target_transform(filtered_df[ACSPublicCoverage.target]).astype(int)
        bld = BinaryLabelDataset(df=filtered_df[:30000], label_names=[tasks[task].target], protected_attribute_names=[tasks[task].group])
        return bld.split(1, shuffle=True, seed=seed)[0]

    features, label, group = tasks[task].df_to_numpy(acs_data)
    X_train, X_test, y_train, y_test, group_train, group_test = train_test_split(
    features[:30000], label[:30000], group[:30000], test_size=0.666666667, shuffle=True, random_state=seed)

    if verb:
        print("train test splits made")

    classifier = _get_model(algo, seed=seed)
    model = make_pipeline(StandardScaler(), classifier)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)
    ret = pd.DataFrame(columns=['score', 'group', 'label'])
    ret['score'] = probs[:,1]
    ret['group_multi'] = group_test
    ret['label'] = y_test
    ret['group'] = [1 if gp == 1 else 0 for gp in ret.group_multi]

    if verb:
        print("trained on:", len(X_train), "samples")
        print("returning: ", len(ret), "samples")

    return ret
