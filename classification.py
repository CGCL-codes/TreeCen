import argparse
from pyexpat import features
import time
from tkinter import Label
import json
import csv
import numpy as np
import random
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score
import pandas as pd
import os

'''
type_cal function
--------------------
This function is used to calculate TPR, FPR, TNR, FNR, F1, Predicsion, Recall of each type

#Arguments
    pre_type: The list of prediction type of code pairs
    lab_type: The list of ordinary type of code pairs
    chosen_type: The type to be calculate(T1, T2, ST3, MT3, T4, None) 
'''

def type_cal(pre_type, lab_type, chosen_type):
    TP = 0
    FN = 0
    TN = 0
    FP = 0
    for i in range(len(lab_type)):
        if(lab_type[i] == chosen_type):
            if(pre_type[i]==1):
                TP+=1
            else:
                FN+=1
        elif(lab_type[i] == 'None'):
            if(pre_type[i]==1):
                FP+=1
            else:
                TN+=1
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    TNR = TN / (TN + FP)
    FNR = FN / (TP + FN)    
    Recall = TP/(TP+FN)
    Precision = TP/(TP+FP)
    F1 = 2 * Precision * Recall / (Precision + Recall)
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    return [chosen_type,F1,Precision,Recall,Accuracy,TPR,FPR,TNR,FNR]

'''
feature_extraction function
--------------------
This function is used to extract the IDs, features, Labels(0:Non-clone; 1:Clone), and Types of clone pairs.

#Arguments
    clone_csv: The .csv file path of clone pairs
    mtx_file: The file that contains six centralities of source codes
    clone_type: The type of clone pairs (T1, T2, ST3, MT3, T4)
    cent_type: The type of centrality 

'''
def feature_extraction_clone(clone_csv, mtx_file, clone_type, cent_type):
    features = []
    Labels = []
    types = []
    ID1 = []
    ID2 = []
    if clone_type == 'ST3':
        clone_type_list = ['ST3', 'VST3']
    else:
        clone_type_list = [clone_type]
    reader_c = csv.reader(open(clone_csv, 'r'))
    for r in reader_c:
        if r[2] in clone_type_list:
            ID1.append(r[0])
            ID2.append(r[1])
            vec_pair = list()
            vec1 = mtx_file[r[0]][cent_type]
            vec2 = mtx_file[r[1]][cent_type]
            vec_pair = vec1 + vec2                            
            features.append(vec_pair)
            Labels.append(1)
            types.append(r[2])
    return ID1,ID2,features,Labels,types

'''
feature_extraction_none function
--------------------
This function is used to extract the IDs, features, Labels(0:Non-clone; 1:Clone) and types of non_clone pairs 

#Arguments
    none_clone_csv: The .csv file path of non_clone pair 
    mtx_file: The file that contains six centralities of source codes
    cent_type: The type of centrality 

'''
def feature_extraction_none(non_clone_csv, mtx_file, cent_type):
    features = []
    Labels = []
    types = []
    ID1 = []
    ID2 = []
    reader_c = csv.reader(open(non_clone_csv, 'r'))
    for r in reader_c:
        if r[2]=='None':
            ID1.append(r[0])
            ID2.append(r[1])
            vec_pair = list()
            vec1 = mtx_file[r[0]][cent_type]
            vec2 = mtx_file[r[1]][cent_type]
            vec_pair = vec1 + vec2                            
            features.append(vec_pair)
            Labels.append(0)
            types.append(r[2])
    return ID1,ID2,features,Labels,types

from sklearn.neighbors import KNeighborsClassifier
def knn_1(vectors, labels):
    X = np.array(vectors)
    Y = np.array(labels)
    kf = KFold(n_splits=10)
    F1s = []
    Precisions = []
    Recalls = []
    Accuracys = []
    TPRs = []
    FPRs = []
    TNRs = []
    FNRs = []
    all_types_pre = []
    all_types_lab = []
    for train_index, test_index in kf.split(X):
        train_X, train_Y = X[train_index], Y[train_index]
        test_X, test_Y = X[test_index], Y[test_index]
        ###data for train ###
        clf = KNeighborsClassifier(n_neighbors=1) 
        clf.fit(train_X, train_Y)
        y_pred = clf.predict(test_X) 
        all_types_pre.extend(y_pred)
        all_types_lab.extend(test_Y)
        
        f1 = f1_score(y_true=test_Y, y_pred=y_pred)
        precision = precision_score(y_true=test_Y, y_pred=y_pred)
        recall = recall_score(y_true=test_Y, y_pred=y_pred)
        accuracy = accuracy_score(y_true=test_Y, y_pred=y_pred)

        TP = np.sum(np.multiply(test_Y, y_pred))
        FP = np.sum(np.logical_and(np.equal(test_Y, 0), np.equal(y_pred, 1)))
        FN = np.sum(np.logical_and(np.equal(test_Y, 1), np.equal(y_pred, 0)))
        TN = np.sum(np.logical_and(np.equal(test_Y, 0), np.equal(y_pred, 0)))

        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        TNR = TN / (TN + FP)
        FNR = FN / (TP + FN)

        F1s.append(f1)
        Precisions.append(precision)
        Recalls.append(recall)
        Accuracys.append(accuracy)
        TPRs.append(TPR)
        FPRs.append(FPR)
        TNRs.append(TNR)
        FNRs.append(FNR)
    print('--------> KNN1:')
    print(F1s, FPRs)
    return ['KNN1', np.mean(F1s), np.mean(Precisions), np.mean(Recalls), np.mean(Accuracys), np.mean(TPRs), np.mean(FPRs), np.mean(TNRs), np.mean(FNRs)],all_types_lab,all_types_pre

def knn_3(vectors, labels):
    X = np.array(vectors)
    Y = np.array(labels)
    kf = KFold(n_splits=10)
    F1s = []
    Precisions = []
    Recalls = []
    Accuracys = []
    TPRs = []
    FPRs = []
    TNRs = []
    FNRs = []
    all_types_pre = []
    all_types_lab = []
    for train_index, test_index in kf.split(X):
        train_X, train_Y = X[train_index], Y[train_index]
        test_X, test_Y= X[test_index], Y[test_index]

        clf = KNeighborsClassifier(n_neighbors=3)
        clf.fit(train_X, train_Y)
        y_pred = clf.predict(test_X)
        all_types_pre.extend(y_pred)
        all_types_lab.extend(test_Y)

        f1 = f1_score(y_true=test_Y, y_pred=y_pred)
        precision = precision_score(y_true=test_Y, y_pred=y_pred)
        recall = recall_score(y_true=test_Y, y_pred=y_pred)
        accuracy = accuracy_score(y_true=test_Y, y_pred=y_pred)

        TP = np.sum(np.multiply(test_Y, y_pred))
        FP = np.sum(np.logical_and(np.equal(test_Y, 0), np.equal(y_pred, 1)))
        FN = np.sum(np.logical_and(np.equal(test_Y, 1), np.equal(y_pred, 0)))
        TN = np.sum(np.logical_and(np.equal(test_Y, 0), np.equal(y_pred, 0)))

        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        TNR = TN / (TN + FP)
        FNR = FN / (TP + FN)

        F1s.append(f1)
        Precisions.append(precision)
        Recalls.append(recall)
        Accuracys.append(accuracy)
        TPRs.append(TPR)
        FPRs.append(FPR)
        TNRs.append(TNR)
        FNRs.append(FNR)
    print('--------> KNN3:')
    print(F1s, FPRs)
    return ['KNN3', np.mean(F1s), np.mean(Precisions), np.mean(Recalls), np.mean(Accuracys), np.mean(TPRs), np.mean(FPRs), np.mean(TNRs), np.mean(FNRs)],all_types_lab,all_types_pre

from sklearn.ensemble import RandomForestClassifier
def randomforest(vectors, labels):
    X = np.array(vectors)
    Y = np.array(labels)
    kf = KFold(n_splits=10)
    F1s = []
    Precisions = []
    Recalls = []
    Accuracys = []
    TPRs = []
    FPRs = []
    TNRs = []
    FNRs = []
    all_types_pre = []
    all_types_lab = []
    for train_index, test_index in kf.split(X):
        train_X, train_Y = X[train_index], Y[train_index]
        test_X, test_Y= X[test_index], Y[test_index]

        clf = RandomForestClassifier(max_depth=8, random_state=0)
        clf.fit(train_X, train_Y)
        y_pred = clf.predict(test_X)
        all_types_pre.extend(y_pred)
        all_types_lab.extend(test_Y)

        f1 = f1_score(y_true=test_Y, y_pred=y_pred)
        precision = precision_score(y_true=test_Y, y_pred=y_pred)
        recall = recall_score(y_true=test_Y, y_pred=y_pred)
        accuracy = accuracy_score(y_true=test_Y, y_pred=y_pred)

        TP = np.sum(np.multiply(test_Y, y_pred))
        FP = np.sum(np.logical_and(np.equal(test_Y, 0), np.equal(y_pred, 1)))
        FN = np.sum(np.logical_and(np.equal(test_Y, 1), np.equal(y_pred, 0)))
        TN = np.sum(np.logical_and(np.equal(test_Y, 0), np.equal(y_pred, 0)))

        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        TNR = TN / (TN + FP)
        FNR = FN / (TP + FN)

        F1s.append(f1)
        Precisions.append(precision)
        Recalls.append(recall)
        Accuracys.append(accuracy)
        TPRs.append(TPR)
        FPRs.append(FPR)
        TNRs.append(TNR)
        FNRs.append(FNR)
    print('--------> randomforest:')
    print(F1s, FPRs)
    return ['randomforest', np.mean(F1s), np.mean(Precisions), np.mean(Recalls), np.mean(Accuracys), np.mean(TPRs), np.mean(FPRs), np.mean(TNRs), np.mean(FNRs)],all_types_lab,all_types_pre

from sklearn.svm import SVC
def svm(vectors, labels):
    X = np.array(vectors)
    Y = np.array(labels)
    kf = KFold(n_splits=10)
    F1s = []
    Precisions = []
    Recalls = []
    Accuracys = []
    TPRs = []
    FPRs = []
    TNRs = []
    FNRs = []
    all_types_pre = []
    all_types_lab = []
    for train_index, test_index in kf.split(X):
        train_X, train_Y = X[train_index], Y[train_index]
        test_X, test_Y = X[test_index], Y[test_index]

        clf = SVC(kernel='linear')
        clf.fit(train_X, train_Y)
        y_pred = clf.predict(test_X)
        all_types_pre.extend(y_pred)
        all_types_lab.extend(test_Y)

        f1 = f1_score(y_true=test_Y, y_pred=y_pred)
        precision = precision_score(y_true=test_Y, y_pred=y_pred)
        recall = recall_score(y_true=test_Y, y_pred=y_pred)
        accuracy = accuracy_score(y_true=test_Y, y_pred=y_pred)

        TP = np.sum(np.multiply(test_Y, y_pred))
        FP = np.sum(np.logical_and(np.equal(test_Y, 0), np.equal(y_pred, 1)))
        FN = np.sum(np.logical_and(np.equal(test_Y, 1), np.equal(y_pred, 0)))
        TN = np.sum(np.logical_and(np.equal(test_Y, 0), np.equal(y_pred, 0)))

        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        TNR = TN / (TN + FP)
        FNR = FN / (TP + FN)

        F1s.append(f1)
        Precisions.append(precision)
        Recalls.append(recall)
        Accuracys.append(accuracy)
        TPRs.append(TPR)
        FPRs.append(FPR)
        TNRs.append(TNR)
        FNRs.append(FNR)
    print('--------> svm')
    print(F1s, FPRs)
    return ['SVM', np.mean(F1s), np.mean(Precisions), np.mean(Recalls), np.mean(Accuracys), np.mean(TPRs), np.mean(FPRs), np.mean(TNRs), np.mean(FNRs)],all_types_lab,all_types_pre

from sklearn import tree
def decision_tree(vectors, labels):
    X = np.array(vectors)
    Y = np.array(labels)
    kf = KFold(n_splits=10)
    F1s = []
    Precisions = []
    Recalls = []
    Accuracys = []
    TPRs = []
    FPRs = []
    TNRs = []
    FNRs = []
    all_types_pre = []
    all_types_lab = []
    for train_index, test_index in kf.split(X):
        train_X, train_Y = X[train_index], Y[train_index]
        test_X, test_Y= X[test_index], Y[test_index]

        clf = tree.DecisionTreeClassifier()
        clf.fit(train_X, train_Y)
        y_pred = clf.predict(test_X)
        all_types_pre.extend(y_pred)
        all_types_lab.extend(test_Y)

        f1 = f1_score(y_true=test_Y, y_pred=y_pred)
        precision = precision_score(y_true=test_Y, y_pred=y_pred)
        recall = recall_score(y_true=test_Y, y_pred=y_pred)
        accuracy = accuracy_score(y_true=test_Y, y_pred=y_pred)

        TP = np.sum(np.multiply(test_Y, y_pred))
        FP = np.sum(np.logical_and(np.equal(test_Y, 0), np.equal(y_pred, 1)))
        FN = np.sum(np.logical_and(np.equal(test_Y, 1), np.equal(y_pred, 0)))
        TN = np.sum(np.logical_and(np.equal(test_Y, 0), np.equal(y_pred, 0)))

        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        TNR = TN / (TN + FP)
        FNR = FN / (TP + FN)

        F1s.append(f1)
        Precisions.append(precision)
        Recalls.append(recall)
        Accuracys.append(accuracy)
        TPRs.append(TPR)
        FPRs.append(FPR)
        TNRs.append(TNR)
        FNRs.append(FNR)
    print('--------> decision_tree')
    print(F1s, FPRs)
    return ['Decision_tree', np.mean(F1s), np.mean(Precisions), np.mean(Recalls), np.mean(Accuracys), np.mean(TPRs), np.mean(FPRs), np.mean(TNRs), np.mean(FNRs)],all_types_lab,all_types_pre

from sklearn.naive_bayes import GaussianNB
def naive_bayes(vectors, labels):
    X = np.array(vectors)
    Y = np.array(labels)
    kf = KFold(n_splits=10)
    F1s = []
    Precisions = []
    Recalls = []
    Accuracys = []
    TPRs = []
    FPRs = []
    TNRs = []
    FNRs = []
    all_types_pre = []
    all_types_lab = []
    for train_index, test_index in kf.split(X):
        train_X, train_Y = X[train_index], Y[train_index]
        test_X, test_Y = X[test_index], Y[test_index]

        clf = GaussianNB()
        clf.fit(train_X, train_Y)
        y_pred = clf.predict(test_X)
        all_types_pre.extend(y_pred)
        all_types_lab.extend(test_Y)

        f1 = f1_score(y_true=test_Y, y_pred=y_pred)
        precision = precision_score(y_true=test_Y, y_pred=y_pred)
        recall = recall_score(y_true=test_Y, y_pred=y_pred)
        accuracy = accuracy_score(y_true=test_Y, y_pred=y_pred)

        TP = np.sum(np.multiply(test_Y, y_pred))
        FP = np.sum(np.logical_and(np.equal(test_Y, 0), np.equal(y_pred, 1)))
        FN = np.sum(np.logical_and(np.equal(test_Y, 1), np.equal(y_pred, 0)))
        TN = np.sum(np.logical_and(np.equal(test_Y, 0), np.equal(y_pred, 0)))

        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        TNR = TN / (TN + FP)
        FNR = FN / (TP + FN)

        F1s.append(f1)
        Precisions.append(precision)
        Recalls.append(recall)
        Accuracys.append(accuracy)
        TPRs.append(TPR)
        FPRs.append(FPR)
        TNRs.append(TNR)
        FNRs.append(FNR)
    print('--------> Naive Bayes')
    print(F1s, FPRs)
    return ['Naive Bayes', np.mean(F1s), np.mean(Precisions), np.mean(Recalls), np.mean(Accuracys), np.mean(TPRs), np.mean(FPRs), np.mean(TNRs), np.mean(FNRs)],all_types_lab,all_types_pre


from sklearn.linear_model import LogisticRegression
def logistic_regression(vectors, labels):
    X = np.array(vectors)
    Y = np.array(labels)
    kf = KFold(n_splits=10)
    F1s = []
    Precisions = []
    Recalls = []
    Accuracys = []
    TPRs = []
    FPRs = []
    TNRs = []
    FNRs = []
    all_types_pre = []
    all_types_lab = []
    for train_index, test_index in kf.split(X):
        train_X, train_Y = X[train_index], Y[train_index]
        test_X, test_Y  = X[test_index], Y[test_index]

        clf = LogisticRegression()
        clf.fit(train_X, train_Y)
        y_pred = clf.predict(test_X)
        all_types_pre.extend(y_pred)
        all_types_lab.extend(test_Y)

        f1 = f1_score(y_true=test_Y, y_pred=y_pred)
        precision = precision_score(y_true=test_Y, y_pred=y_pred)
        recall = recall_score(y_true=test_Y, y_pred=y_pred)
        accuracy = accuracy_score(y_true=test_Y, y_pred=y_pred)

        TP = np.sum(np.multiply(test_Y, y_pred))
        FP = np.sum(np.logical_and(np.equal(test_Y, 0), np.equal(y_pred, 1)))
        FN = np.sum(np.logical_and(np.equal(test_Y, 1), np.equal(y_pred, 0)))
        TN = np.sum(np.logical_and(np.equal(test_Y, 0), np.equal(y_pred, 0)))

        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        TNR = TN / (TN + FP)
        FNR = FN / (TP + FN)

        F1s.append(f1)
        Precisions.append(precision)
        Recalls.append(recall)
        Accuracys.append(accuracy)
        TPRs.append(TPR)
        FPRs.append(FPR)
        TNRs.append(TNR)
        FNRs.append(FNR)
    print('--------> logistic_regression')
    print(F1s, FPRs)
    return ['Logistic_regression', np.mean(F1s), np.mean(Precisions), np.mean(Recalls), np.mean(Accuracys), np.mean(TPRs), np.mean(FPRs), np.mean(TNRs), np.mean(FNRs)],all_types_lab,all_types_pre

'''
random_features function
--------------------
This function is used to randomly disrupt the order of code pairs

#Arguments
    ID1: The list of ids of the first source code
    ID2: The list of ids of the second source code
    vectors: The vectors that extract from the function feature_extraction
    labels: The label of code pairs
    types: The clone type of code pairs(T1, T2, ST3, MT3, T4, None)

'''
def random_features(ID1, ID2, vectors, labels, types):
    Vec_Lab = []
    for i in range(len(vectors)):
        vec = vectors[i]
        lab = labels[i]
        type = types[i]
        id1 = ID1[i]
        id2 = ID2[i]
        vec.append(id1)
        vec.append(id2)
        vec.append(type)
        vec.append(lab)
        Vec_Lab.append(vec)
    random.shuffle(Vec_Lab)
    return [m[-4] for m in Vec_Lab] ,[m[-3] for m in Vec_Lab] ,[m[:-4] for m in Vec_Lab],[m[-1] for m in Vec_Lab],[m[-2] for m in Vec_Lab]#ID1,ID2,features,Labels,types


'''
classification function
--------------------
This function uses 7 machine learning algorithms to train the model and predict the code pairs

#Arguments
    ID1: The list of ids of the first source code
    ID2: The list of ids of the second source code
    vectors: The vectors that extract from the function feature_extraction
    labels: The label of code pairs
    types: The clone type of code pairs(T1, T2, ST3, MT3, T4, None)

'''
def classification(id1,id2,vectors, labels, types):
    ID1,ID2,Vectors, Labels, Types = random_features(id1, id2, vectors, labels, types)
    result_dict = {}
    csv_data = []
    csv_data.append(['ML_Algorithm', 'F1', 'Precision', 'Recall', 'Accuracy', 'TPR', 'FPR', 'TNR', 'FNR'])

    all_res, lab_type, pre_type = knn_1(ID1,ID2,Vectors, Labels, Types)
    csv_data.append(all_res)
    result_dict['knn1'] = [lab_type,pre_type]

    all_res, lab_type, pre_type = knn_3(ID1,ID2,Vectors, Labels, Types)
    csv_data.append(all_res)
    result_dict['knn3'] = [lab_type,pre_type]

    all_res, lab_type, pre_type = randomforest(ID1,ID2,Vectors, Labels, Types)
    csv_data.append(all_res)
    result_dict['randomforest'] = [lab_type,pre_type]
    
    all_res, lab_type, pre_type = naive_bayes(ID1,ID2,Vectors, Labels, Types)
    csv_data.append(all_res)
    result_dict['naive_bayes'] = [lab_type,pre_type]
    
    all_res, lab_type, pre_type = decision_tree(ID1,ID2,Vectors, Labels, Types)
    csv_data.append(all_res)
    result_dict['decision_tree'] = [lab_type,pre_type]
    
    
    all_res, lab_type, pre_type = svm(ID1,ID2,Vectors, Labels, Types)
    csv_data.append(all_res)
    result_dict['svm'] = [lab_type,pre_type]
    
    all_res, lab_type, pre_type = logistic_regression(ID1,ID2,Vectors, Labels, Types)
    csv_data.append(all_res)
    result_dict['logistic_regression'] = [lab_type,pre_type]
    return csv_data

def main():
    result_path = '/Users/apple/Desktop/code-clone/result1'
    clone_csv = '/home/CodeClone/dataset/clone_pair.csv'
    none_csv = '/home/CodeClone/dataset/none_clone_pair.csv'
    mtx_file_path = '/home/CodeClone/dataset/cent_matrix.json'
    cent_type_list = ['cent_harm', 'cent_eigen', 'cent_close', 'cent_betwen', 'cent_degree', 'cent_katz']
    clone_type_list = ['T1','T2','ST3','MT3','T4']
    with open(mtx_file_path,'r') as f:
        all_mtxs = json.load(f)
    for cent_type in cent_type_list:
        outpath = os.path.join(result_path,cent_type)
        print(cent_type)
        ID1_n,ID2_n,features_n,Labels_n,types_n = feature_extraction_none(none_csv,all_mtxs,cent_type)
        for clone_type in clone_type_list:
            outpath = outpath + '_clone_type.csv'
            print(clone_type)
            ID1_c,ID2_c,features_c,Labels_c,types_c = feature_extraction_clone(clone_csv,all_mtxs,clone_type,cent_type)
            ID1 = ID1_c+ID1_n
            ID2 = ID2_c+ID2_n
            features = features_c+features_n
            Labels = Labels_c+Labels_n
            types = types_c+types_n
            csv_data = classification(ID1,ID2,features,Labels,types)
            for i in csv_data:
                print(csv_data)
            print("\n")
    print("\n")


if __name__ == '__main__':
    main()