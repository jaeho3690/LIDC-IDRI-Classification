import numpy as np
import torch
import torch.nn.functional as F


def Accuracy(output, label):
    total = 0
    correct =0
    output_prob = (output>0.5).int()
    correct += torch.sum(output_prob==label).item()
    total += len(label)

    return  correct / total


def Confusion_matrix(output, label):
    total = 0
    smooth = 0.1

    output_prob = (output>0.5).int()
    label = label.int()

    conf_matrix = torch.zeros(2, 2)
    for t, p in zip(label, output_prob):
        conf_matrix[t, p] += 1
    TP = conf_matrix[1,1].item()
    TN = conf_matrix[0,0].item()
    FP = conf_matrix[0,1].item()
    FN = conf_matrix[1,0].item()

    total += len(label)
    accuracy = (TP+TN)/total
    sensitivity = TP / (TP+FN+ smooth)
    specificity = TN / (TN+FP+ smooth)

    return  accuracy, sensitivity, specificity

def Confusion_matrix2(output, label):
    total = 0
    smooth = 0.1

    output_prob = (output>0.5).int()
    label = label.int()

    conf_matrix = torch.zeros(2, 2)
    for t, p in zip(label, output_prob):
        conf_matrix[t, p] += 1
    TP = conf_matrix[1,1].item()
    TN = conf_matrix[0,0].item()
    FP = conf_matrix[0,1].item()
    FN = conf_matrix[1,0].item()


    return  TP,TN,FN,FP