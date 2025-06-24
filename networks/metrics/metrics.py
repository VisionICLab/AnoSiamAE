from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import os
import numpy as np

from typing import *

class ADMetrics():
    def __init__(self, epsilon:float=1.0E-6):
        self.epsilon = epsilon

    def compute_metrics(self,
                        predictions:np.array,
                        labels:np.array) -> Union[float, float, float]:
        tp = (predictions*labels).sum()
        tn = ((1-predictions)*(1-labels)).sum()
        fp = (predictions*(1-labels)).sum()
        fn = ((1-predictions)*labels).sum()

        precision = (tp+self.epsilon)/(tp+fp+self.epsilon)
        sensitivity = (tp+self.epsilon)/(tp+fn+self.epsilon)
        specificity = (tn+self.epsilon)/(tn+fp+self.epsilon)

        return float(precision), float(sensitivity), float(specificity)
            
    def image_metrics(self, 
                        scores:np.array,
                        labels:np.array,
                        threshold:float) -> Union[float, float, float]:
        predictions = (scores>threshold).astype(np.int64)
        return self.compute_metrics(predictions, labels)
    
    def roc(self, 
            scores:np.array, 
            labels: np.array,
            metric:str,
            plot:bool=True, 
            output:str=None) -> Union[float, float]:
        # Compute ROC
        fpr, tpr, thresholds = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)

        # Geometric mean between sensitivity (True Positive rate) and specificity (False Positive rate)
        gmeans = np.sqrt(tpr*(1-fpr))
        idx = np.argmax(gmeans)
        thresh = thresholds[idx]
        if plot:
            plt.figure()
            plt.plot([0,1], [0,1], linestyle='--', label='baseline')
            plt.plot(fpr, tpr, marker=".", label="{}: (ROC_AUC={:.2f})".format(metric, roc_auc))
            plt.scatter(fpr[idx], tpr[idx], marker="o", color="red", label="Best (GMean={:.2f})".format(gmeans[idx]))
            plt.title("ROC Curve")
            plt.xlabel("False Positive rate")
            plt.ylabel("True Positive rate")
            plt.legend()
            if output!=None:
                if not os.path.exists(output):
                    os.makedirs(output)
                plt.savefig(os.path.join(output, f"ROCCurve_{metric}.png"))
            plt.clf()
            plt.close()

        return float(roc_auc), float(thresh)

    def pr(self, 
            scores:np.array, 
            labels: np.array,
            metric:str,
            plot:bool=True, 
            output:str=None) -> Union[float, float]:
        # Compute Precision-Recall curve
        prec, rec, thresholds = precision_recall_curve(labels, scores)
        pr_auc = auc(rec, prec)
        
        # Find the best threshold
        f1 = np.divide(2*prec*rec, prec+rec, out=np.zeros_like(prec), where=prec+rec!=0)
        idx = np.argmax(f1)
        thresh = thresholds[idx]
        if plot:
            plt.figure()
            plt.plot(rec, prec, marker=".", label="{}: (PR_AUC={:.2f})".format(metric, pr_auc))
            plt.scatter(rec[idx], prec[idx], marker="o", color="red", label="Best (F1={:.2f})".format(f1[idx]))
            plt.title("Precision-Recall Curve")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.legend()
            if output!=None:
                if not os.path.exists(output):
                    os.makedirs(output)
                plt.savefig(os.path.join(output, f"PRCurve_{metric}.png"))
            plt.clf()
            plt.close()

        return float(pr_auc), float(thresh)

    def __call__(self, 
                 metric:str,
                 scores:np.array, 
                 labels: np.array, 
                 plot:bool=True, 
                 output:str=None) -> dict:
        roc = {}
        roc["auc"], roc["threshold"] = self.roc(scores, labels, metric, plot, output)
        roc["image_prec"], roc["image_sens"], roc["image_spec"] = self.image_metrics(scores, labels, roc["threshold"])

        pr = {}
        pr["auc"], pr["threshold"] = self.pr(scores, labels, metric, plot, output)
        pr["image_prec"], pr["image_sens"], pr["image_spec"] = self.image_metrics(scores, labels, pr["threshold"])

        return {"roc":roc, "pr":pr}
