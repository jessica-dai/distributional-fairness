import numpy as np
import pandas as pd 

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, normalize
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from folktables import ACSDataSource, ACSIncome, ACSPublicCoverage, ACSMobility, ACSEmployment, BasicProblem

def get_data(dataset, seed, verb):
    if dataset == 'taiwan':
        return gen_taiwan_credit(seed=seed, verb=verb)
    elif dataset == 'german':
        return gen_german_probs(seed=seed, verb=verb)
    elif dataset == 'adult_old':
        return gen_adult_probs(seed=seed, verb=verb)
    elif dataset == 'adult_new':
        return gen_new_adult(seed=seed, verb=verb, task='income')
    elif dataset == 'public':
        return gen_new_adult(seed=seed, verb=verb, task='public')
    elif dataset == 'employment':
        return gen_new_adult(seed=seed, verb=verb, task='employment_sex')

    else: 
        print("dataset typo?")

def gen_taiwan_credit(seed=0, verb=False):
    credit = pd.read_csv('data/UCI_Credit_Card.csv') # 30k rows

    credit['EDUCATION'] = credit['EDUCATION'] < 2
    labels = credit['default.payment.next.month'].values
    X = credit.drop(columns=['default.payment.next.month'])
    sens = credit['EDUCATION'].values

    X_train, X_test, y_train, y_test, group_train, group_test = train_test_split(
        X, labels, sens, test_size=0.4, random_state=seed)

    model = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=seed))
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
    german = pd.read_csv('data/german.csv') # from friedler 2019 preprocessing
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

def gen_adult_probs(seed=0, verb=False, sens='sex'):
    adult = pd.read_csv('data/adult_numerical.csv') # from friedler 2019 preprocessing
    # dataset wrangling - total 30162 
    X = adult.drop(columns=['income-per-year', 'race-sex'])
    X['race'] = np.array([1 if a == 'White' else 0 for a in X.race])
    X['sex'] = np.array([1 if a == 'Male' else 0 for a in X.sex])
    sens = X[sens]
    labels = np.ones(len(X))
    labels[adult['income-per-year'] == '<=50K'] = 0
    X = X.values

    good_inds = ~np.isnan(X).any(axis=1) # all inds are good
    X = X[good_inds]
    labels = labels[good_inds]
    sens = sens[good_inds].values

    # scale and shuffle
    ss = MinMaxScaler().fit(X)
    X = normalize(ss.transform(X))

    np.random.seed(seed)
    shuffleinds = np.arange(len(X))
    np.random.shuffle(shuffleinds)

    X = X[shuffleinds]
    labels = labels[shuffleinds]
    sens = sens[shuffleinds]

    # train and predict 
    split_ind = 15000
    classifier = RandomForestClassifier(random_state=seed).fit(X[:split_ind], labels[:split_ind])

    probs = classifier.predict_proba(X[split_ind:])
    
    ret = pd.DataFrame(columns=['score', 'group', 'label'])
    ret['score'] = probs[:,1]
    ret['label'] = labels[split_ind:]
    ret['group'] = sens[split_ind:]

    if verb:
        print("trained on:", split_ind, "samples")
        print("returning: ", len(ret), "samples")

    return ret

def _get_employment_sex():
    return BasicProblem(features=ACSEmployment.features, 
                                 target=ACSEmployment.target,
                                 group='SEX',
                                 group_transform=lambda x: x > 1,
                                 target_transform=ACSEmployment.target_transform,
                                 postprocess=lambda x: np.nan_to_num(x, -1))

def gen_new_adult(seed, verb, task='income'):

    tasks = {
        'income': ACSIncome,
        'public': ACSPublicCoverage,
        'mobility': ACSMobility,
        'employment_sex': _get_employment_sex()
    }
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=['CA'], download=True)
    features, label, group = tasks[task].df_to_numpy(acs_data)
    X_train, X_test, y_train, y_test, group_train, group_test = train_test_split(
    features[:30000], label[:30000], group[:30000], test_size=0.4, random_state=seed)

    model = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=seed))
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
