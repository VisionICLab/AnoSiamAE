from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import os
import numpy as np

from PIL import Image
from typing import *

def ROCCurve(labels, X, metric, level="image", plot= True, output=None):
    """
        ROCCurve computation
    """

    fpr, tpr, thresholds = roc_curve(np.array(labels)>=1, X)
    roc_auc = auc(fpr, tpr)
    
    # Geometric mean between sensitivity (True Positive rate) and specificity (False Positive rate)
    # Find the best threshold
    gmeans = np.sqrt(tpr*(1-fpr)) 
    idx = np.argmax(gmeans)
    sens = tpr[idx]
    spec = 1-fpr[idx]
    if plot:
        plt.figure()
        plt.plot([0,1], [0,1], linestyle='--', label='baseline')
        plt.plot(fpr, tpr, marker=".", label="{}: (AUC={:.2f})".format(metric, roc_auc))
        plt.scatter(fpr[idx], sens, marker="o", color="red", label="Best (GMean={:.2f})".format(gmeans[idx]))
        plt.title("ROC Curve")
        plt.xlabel("False Positive rate")
        plt.ylabel("True Positive rate")
        plt.legend()
        if output!=None:
            if not os.path.exists(output):
                os.makedirs(output)
            plt.savefig(os.path.join(output, f"ROCCurve_{metric}_{level}.png"))
        plt.clf()
        plt.close()
    return roc_auc, thresholds[idx], sens, spec, np.max(gmeans)


def PRCurve(labels, X, metric, level="image", plot=True, output=None):
    """
        Precision-Recall Curve computation
    """
    prec, rec, thresholds = precision_recall_curve(np.array(labels)>=1, X)
    ap = average_precision_score(np.array(labels)>=1, X)
    
    # Find the best threshold
    f1 = np.divide(2*prec*rec, prec+rec, out=np.zeros_like(prec), where=prec+rec!=0)
    idx = np.argmax(f1)
    if plot:
        plt.figure()
        plt.plot(rec, prec, marker=".", label="{}: (AP={:.2f})".format(metric, ap))
        plt.scatter(rec[idx], prec[idx], marker="o", color="red", label="Best (F1={:.2f})".format(f1[idx]))
        plt.title("Precision-Recall Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        if output!=None:
            if not os.path.exists(output):
                os.makedirs(output)
            plt.savefig(os.path.join(output, f"PRCurve_{metric}_{level}.png"))
        plt.clf()
        plt.close()
    return ap, thresholds[idx], prec[idx], rec[idx]

def DensityHist(metric, norm, abn, threshold=None, level="image", bins=50, save=False, output=None):
    fig = plt.figure()
    plt.hist(norm, bins=bins, density=False, histtype='barstacked', alpha=0.5, color = "g", label="Normal data", weights=np.ones(len(norm)) / len(norm))       
    plt.hist(abn, bins=bins, density=False, histtype='barstacked', alpha=0.5, color = "r", label="Abnormal data", weights=np.ones(len(abn)) / len(abn))   
    if threshold!=None:
        plt.axvline(threshold, color='b', linestyle='dashed', linewidth=1, label="gmean_thresh={:.2f}".format(threshold))
    plt.title(f"Probability Density Distribution | {metric}")
    plt.xlabel("Anomaly score")
    plt.ylabel("Density")
    plt.legend()
    fig.canvas.draw()
    img = Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
    if output!=None and save:
        if not os.path.exists(output):
            os.makedirs(output)
        plt.savefig(os.path.join(output, f"PDD_{metric}_{level}.png"))
    plt.clf()
    plt.close()
    return img