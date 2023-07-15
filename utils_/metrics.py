from sklearn.metrics import roc_auc_score, accuracy_score, multilabel_confusion_matrix
import numpy as np

def auc(all_tar, all_output):
    res=roc_auc_score(all_tar, all_output, multi_class='ovo')
    return round(res,4)
def acc(all_tar, all_output):
    res =accuracy_score(all_tar, all_output)
    return round(res,4)

def auc_sens(all_tar,all_output):
    res=roc_auc_score(all_tar, all_output)
    return round(res,4)