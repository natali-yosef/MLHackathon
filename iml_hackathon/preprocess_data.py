import math

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingRegressor
from sklearn.linear_model import RidgeClassifierCV
import plotly
from plotly import graph_objects as go
import sys
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_validate

import numpy as np
import pandas as pd
import re
from dateutil.parser import parse
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def is_date(string, fuzzy=False):
    """
    Return whether the string can be interpreted as a date.

    :param string: str, string to check for date
    :param fuzzy: bool, ignore unknown tokens in string if True
    """
    try:
        parse(string, fuzzy=fuzzy)
        return True
    except Exception:
        return False


# possible labels of cancer areas
y_labels = np.array(['LYM - Lymph nodes', 'BON - Bones', 'HEP - Hepatic', 'PUL - Pulmonary',
            'SKI - Skin', 'PER - Peritoneum', 'OTH - Other',
            'PLE - Pluera', 'ADR - Adrenals', 'BRA - Brain', 'MAR - Bone Marrow'])


def her2_categorical(data: str):
    neg_literals = {'neg', 'NEG', 'Neg', 'eg', 'שלילי', 'akhkh', 'akhah'}
    pos_literals = {'pos', 'POS', 'Pos', 'חיובי', 'jhuch'}
    uncertain_values = {'uncertain', 'equivocal', '?', 'borderline'}
    neg_values = {'1', '0', '-'}
    uncertain_second_check = {'2'}
    pos_values = {'3', '+'}
    if any([neg in data.lower() for neg in neg_literals]):
        return 'negative'
    elif any([pos in data.lower() for pos in pos_literals]):
        return 'positive'
    elif any([uc in data.lower() for uc in uncertain_values]):
        return 'uncertain'
    elif any([x in data.lower() for x in neg_values]):
        return 'negative'
    elif any([x in data.lower() for x in uncertain_second_check]):
        return 'uncertain'
    elif any([x in data.lower() for x in pos_values]):
        return 'positive'
    else:
        return 'uncertain'


def histo_degree(val: str):
    if ('1' in val):
        return 1
    if ('2' in val):
        return 2
    if '3' in val:
        return 3
    if '4' in val:
        return 4
    if 'X' in val:
        return 2.5
    if 'null' in val.lower():
        return 2.5


def KI67_protein(val: str):
    if val.isdigit():
        return int(val)
    if (is_date(val)):
        return 0
    li = list(map(int, re.findall(r'\d+', val)))
    if li == [] or 100 < max(li):
        return 0
    return max(li)


feature_preprocess_funcs = {'Her2': her2_categorical, 'Histopatological degree': histo_degree, 'KI67 protein': KI67_protein}
first_label_name = 'אבחנה-Location of distal metastases'
second_label_name = 'אבחנה-Tumor size'


def split_labels(labels_df):
    # first_label_name = labels_df.columns[0]
    for label in y_labels:
        labels_df[label] = ""
    first_arr = labels_df[first_label_name].to_numpy(dtype=str)
    for y_label in y_labels:
        if y_label not in labels_df.columns:
            print(y_label)
        labels_df[y_label] = np.where(np.char.find(first_arr, y_label) != -1, 1, 0)
    # drop the first row - original row
    return labels_df.drop(columns=first_label_name, axis=1)


def replace_zeros_with_mean(df, colname):
    m = np.median(df[colname].loc[df[colname] != 0])
    df[colname].loc[df[colname] == 0] = m


def load_data(filename: str, label_path: str, question=1, train=True):
    full_data = pd.read_csv(filename)
    labels = pd.DataFrame()
    label_titles = []
    if train:
        labels = pd.read_csv(label_path)
        if question == 1:
            labels = split_labels(labels)
        labels.columns = labels.columns.str.replace('אבחנה-', '')
        label_titles = labels.columns

    total_data = pd.concat([full_data, labels], axis=1)
    total_data = total_data.drop(columns=[' Form Name', 'User Name'])
    # 20 features remain
    if train:
        original_cols = total_data.columns
        # total_data = total_data.drop_duplicates(subset=original_cols, keep='last')

    merged_data = total_data
    # merged_data = merge_same_patient(pd.concat([full_data, labels], axis=1))
    merged_data.columns = merged_data.columns.str.replace('אבחנה-', '')

    for feature in merged_data.columns:
        if feature in feature_preprocess_funcs:
            new_vec = merged_data[feature].astype('str').apply(feature_preprocess_funcs[feature])
            merged_data[feature] = new_vec

    columns_to_remove = {'Tumor width', 'Tumor depth', 'Diagnosis date',
                         'Surgery date1', 'Surgery date2', 'Surgery date3', 'surgery before or after-Activity date', 'pr',
                         'er', 'id-hushed_internalpatientid', 'Nodes exam', 'Positive nodes'}

    non_categorical = {'Age', 'Diagnosis date', 'KI67 protein', 'Histopatological degree', 'KI67 protein',
                       'id-hushed_internalpatientid', 'pr', 'er', 'Nodes exam', 'Positive nodes'}.union(set(label_titles))
    categorical_features = list(set(merged_data.columns) - columns_to_remove - non_categorical)
    data_with_dummies = pd.get_dummies(merged_data, columns=categorical_features, prefix=categorical_features,
                                       drop_first=True)
    # for col_name in {"KI67 protein"}:
    #     replace_zeros_with_mean(data_with_dummies, col_name)
    # data_with_dummies.drop(columns=categorical_features, axis=1, inplace=True)
    data_with_dummies = data_with_dummies.drop(columns=list(columns_to_remove), axis=1)
    cols = list(set(data_with_dummies.columns) - set(label_titles))
    return data_with_dummies[cols], data_with_dummies[list(label_titles)]


def merge_same_patient(total_data):
    total_data = total_data.drop(columns=[' Form Name', 'User Name'])
    original_cols = total_data.columns
    return total_data.drop_duplicates(subset=original_cols, keep='last')


def predict_size_choose_model(datapath, labelpath):
    y_labels = 'Tumor size'
    df, response = load_data(datapath, labelpath, question=2)
    X_cols = set(df.columns) - {'id-hushed_internalpatientid', 'pr', 'er', 'Nodes exam', 'Positive nodes',
                                'Diagnosis date'} - set(y_labels)

    trainX, testX, trainY, testY = train_test_split(df[X_cols], response, test_size=0.2)

    for col_name in {"KI67 protein"}:
        replace_zeros_with_mean(trainX, col_name)
        replace_zeros_with_mean(testX, col_name)

    validation_vec = []

    x_axis = np.linspace(0.0001, 50, 100)
    for alpha in x_axis:
        cv_results = cross_validate(Ridge(alpha=alpha), X=trainX, y=trainY,
                                   scoring='neg_mean_squared_error', return_train_score=True)
        validation_vec.append([np.mean(cv_results['test_score']), np.mean(cv_results['train_score'])])

    validation_vec = np.abs(np.array(validation_vec))

    fig = go.Figure()
    fig.add_traces([go.Scatter(x=x_axis, y=validation_vec[:,0], mode='lines',
                               line=dict(color='blue'),
                               name='validation error'),
                    go.Scatter(x=x_axis, y=validation_vec[:, 1], mode='lines',
                               line=dict(color='green'),
                               name='train error'),
                    ]).update_layout(title='Cross-validation of regularization values with ridge regression',
                                     xaxis_title= 'alpha',
                                     yaxis_title= 'MSE'
                                     ).show()

    best_alpha = x_axis[np.argmin(validation_vec[:, 0])]
    model = Ridge(alpha=best_alpha)
    model.fit(trainX, trainY)
    test_loss = sklearn.metrics.mean_squared_error(model.predict(testX), testY)
    train_loss = sklearn.metrics.mean_squared_error(model.predict(trainX), trainY)
    print(f"best alpha={best_alpha}")
    print(f"test loss of linear regression: {test_loss}")
    print(f"train loss of linear regression: {train_loss}")
    r1 = Ridge(alpha=best_alpha)
    r2 = RandomForestRegressor(n_estimators=50, max_depth=30, random_state=5)
    r3 = KNeighborsRegressor(n_neighbors=5)
    
    ensemble = VotingRegressor([('lr', r1), ('rf', r2), ('r3', r3)])
    ensemble.fit(trainX, trainY)
    test_loss = sklearn.metrics.mean_squared_error(np.abs(ensemble.predict(testX)), testY)
    train_loss = sklearn.metrics.mean_squared_error(np.abs(ensemble.predict(trainX)), trainY)
    print(f"test loss of ensemble: {test_loss}")
    print(f"train loss of ensemble: {train_loss}")
    
    ensemble.fit(df[X_cols], response)
    return ensemble


def build_model(data, labels):
    labels['HasMetastases'] = labels[labels.columns.tolist()].sum(axis=1)
    labels['NoMetastases'] = np.where(labels['HasMetastases'] > 0, 0, 1)

    labels['HasMetastases'] = np.where(labels['HasMetastases'] > 0, 1, 0)
    trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.25)

    ##################################################
    ##        KNeighborsClassifier on train          #
    ##################################################
    knn_model = KNeighborsClassifier(n_neighbors=2)
    knn_model.fit(trainX, trainY.drop(columns="HasMetastases", axis=1))
    knn_pred_train = knn_model.predict(testX)
    print("knn score: ", f1_score(testY.drop(columns="HasMetastases", axis=1), knn_pred_train, average='micro'))

    ##################################################
    ##        DecisionTreeClassifier on train        #
    ##################################################

    ridge = RidgeClassifierCV(alphas=np.linspace(0, 2, 100), cv=5)
    ridge.fit(trainX, trainY.drop(columns="HasMetastases", axis=1))
    ridge_model_pred_train = ridge.predict(testX)
    print("ridge_model score: ", f1_score(testY.drop(columns="HasMetastases", axis=1), ridge_model_pred_train, average='micro'))


    ##################################################
    ##        two models on train                    #
    ##################################################

    # logistic_model = LogisticRegression(penalty="none")
    # logistic_model.fit(trainX, trainY['HasMetastases'])
    #
    # tree_model = DecisionTreeClassifier(max_depth=1)
    # prediction = logistic_model.predict(trainX)
    #
    #
    #
    # trainX_has = trainX[labels['HasMetastases'] > 0]
    # trainY_has = trainY[labels['HasMetastases'] > 0].drop(columns="HasMetastases", axis=1)
    # tree_model_complex = DecisionTreeClassifier(max_depth=10)
    #
    # tree_model_complex.fit(trainX_has, trainY_has)
    #
    # predict_first = logistic_model.predict(trainX)
    # trainX_has = trainX[predict_first > 0]
    # trainY_has = trainY[predict_first > 0].drop(columns="HasMetastases", axis=1)
    # predict_second = tree_model_complex.predict(trainX_has)
    # base = np.zeros((trainX.shape[0], 11))
    # base[predict_first > 0, :] = predict_second
    # print(f1_score(trainY.drop(columns="HasMetastases", axis=1), base, average='micro'))


def fit_and_predict_task1(train_data, train_labels, test_data):
    hasMetastases = train_labels[train_labels.columns.tolist()].sum(axis=1)
    train_labels['NoMetastases'] = np.where(hasMetastases > 0, 0, 1)

    # trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.25)

    ##################################################
    ##        KNeighborsClassifier on train          #
    ##################################################
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(train_data, train_labels)

    prediction = knn_model.predict(test_data)[:, :-1]
    # print("score: ", f1_score(test_labels, prediction, average='micro'))
    result_df = pd.DataFrame(columns=[first_label_name])
    for s in prediction:
        if len(np.where(s > 0)[0]) > 0:
            result_df.loc[len(result_df.index)] = [str(list(y_labels[np.where(s > 0)]))]
        else:
            result_df.loc[len(result_df.index)] = '[]'
    result_df.to_csv('pred_task1.csv', index=False)


if __name__ == '__main__':
    np.random.seed(9)
    datapath = sys.argv[1]
    labelpath = sys.argv[2]
    test_path = sys.argv[3]
    q_num = sys.argv[4]
    train_data, train_labels = load_data(datapath, labelpath, int(q_num), True)
    # test_data, test_labels = load_data(datapath, labelpath, q_num, False)

    # match between the features of the test and the features of the train:
    test_feats, _i = load_data(test_path, "", int(q_num), False)
    cols_for_test = set(test_feats.columns) - {'id-hushed_internalpatientid', 'pr', 'er', 'Nodes exam',
                                               'Positive nodes','Diagnosis date'}
    cols_for_train = set(train_data.columns) - {'id-hushed_internalpatientid', 'pr', 'er', 'Nodes exam',
                                                'Positive nodes','Diagnosis date'}
    for col in cols_for_train:
        if col not in cols_for_test:
            test_feats[col] = 0
    test_feats = test_feats[cols_for_train]
    if int(q_num) == 2:
        ensemble = predict_size_choose_model(datapath, labelpath)
        # train_data.align(test_feats, join='inner', axis=1)
        vec = np.abs(ensemble.predict(test_feats))
        output_df = pd.DataFrame(columns=[second_label_name])
        output_df[second_label_name] = vec
        output_df.to_csv("pred_task2.csv", index=False)
    #build_model(data, labels)
    # build_model(train_data, train_labels)
    if int(q_num) == 1:
        fit_and_predict_task1(train_data, train_labels, test_feats)
