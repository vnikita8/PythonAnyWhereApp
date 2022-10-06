# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

def get_MLP_Answer(predict):
    df = pd.read_csv("1 (1).csv", delimiter=';')
    df = df.replace({"Smoking":{"к":"1", "н":"0", " ":"0"}}).replace({'IL1b':{'T/T':1,"T/C":1, "C/C":0}}).replace({'TNF':{'A/A':1,'G/G':0,'G/A':0}})
    df = df.replace({"APEX1": {"G/G":1, "T/T":0,"T/G":0}}).replace({"XPD": {"T/G":1,"G/G":1, "T/T":0}})
    df = df.replace({"EGFR": {"A/A" :1, "T/T":0,"A/T":0}}).replace({"CHEK2":{"N/P":1,"P/P":1, "N/N":0}})
    df = df.replace({"TGFb1": {"G/G":1, "G/C":0,"C/C":0}}).replace({"EPHX1 ": {"T/T":1, "T/C":0,"C/C":0}})
    df = df.dropna()
    y = df['Status'].values
    df = df.drop('Status', axis = 1)
    x = df.values
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)
    mlp = MLPClassifier(alpha=1, max_iter=1000)
    mlp.fit(x_train, y_train)
    y_pred_mlp = mlp.predict(x_test)
    acc_mlp = accuracy_score(y_test, y_pred_mlp)
    x_quest = np.array([predict])
    y_answer = mlp.predict(x_quest)
    return list(acc_mlp, y_answer)

