"""
Created on Wed Oct 24 14:00:25 2018

@author: Mor
"""

from bilateral_model import bilateral_net
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from plot_confusion_matrix import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

right_train = np.load('right_train.npy')
left_train = np.load('left_train.npy')
right_test = np.load('right_test.npy')
left_test = np.load('left_test.npy')
y_train = np.load('y_train.npy')[:,1]
y_test = np.load('y_test.npy')[:,1]

#%%
# Define hyper-parameters
bs = 64; ep = 100; dp = 0.5; lr = 1e-4

bi_net = bilateral_net(dp, bs, ep, lr, y_train, y_test, right_train, right_test,
                 left_train, left_test, plt_loss = 1)
# train the model
#bi_net.train()

# test the model
bi_net.test()
y_pred = bi_net.y_pred

#%%
# Plot confusion matrix
cm = confusion_matrix(y_test, np.around(y_pred))
labels = ['Healthy','Breast Cancer']
plot_confusion_matrix(cm,labels,title='Confusion Matrix - Test Set',normalize=True)
plt.savefig('conf_mat')
plt.show(); plt.close()

# AUC and ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
auc = roc_auc_score(y_test, y_pred)
print('AUC value is:', auc)

plt.plot(fpr, tpr, lw=1, label='ROC curve (auc = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='darkorange', lw=1, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('ROC Curve.png'); 
plt.show(); plt.close()
