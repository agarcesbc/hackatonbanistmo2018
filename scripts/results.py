import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from algo import algoritmos

algoritmo = algoritmos()


#y = np.array([1, 1, 2, 2])
#scores = np.array([0.1, 0.4, 0.35, 0.8])
#fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)
#area = metrics.auc(fpr, tpr)
#plt.figure()
#plt.plot(fpr[1], tpr[1], color='darkorange', label='ROC curve (area = %0.2f)' % area)
#plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.05])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('Receiver operating characteristic example')
#plt.legend(loc="lower right")
#plt.show()