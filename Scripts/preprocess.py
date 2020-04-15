'''

    Python file containing functions and classes to
    clean text and encode it


'''
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import zero_one_loss
from sklearn.metrics import explained_variance_score
import pandas as pd


N = 0.5


def remove_whitespace(text):
    """remove extra whitespaces from text"""
    text = text.strip()
    return " ".join(text.split())

def func_remove_char_specific(text):
    table = '!"#$%&()*+,./:;<=>?@[\]^_`{|}~•'
    table = str.maketrans('', '', table)
    words = text.split()
    stripped = [w.translate(table) for w in words]
    return ' '.join(stripped)

def func_remove_upper_case(text):
    words = text.split()
    stripped = [w.lower() if w.isupper() else w for w in words]
    return " ".join(stripped)

def func_plot_history(history):
    
    plt.figure(figsize=(15,10))
    plt.subplot(221)
    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.grid(True)


    # Plot training & validation loss values
    plt.subplot(222)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.grid(True)
    plt.show()


    
def func_confusion_matrix(model, valid_y, valid_seq_x):
    return pd.DataFrame(confusion_matrix(valid_y, [1 if i>N else 0 for i in model.predict(valid_seq_x)]), \
             columns=['Predicted Negative', "Predicted Positive"], index=['Actual Negative', 'Actual Positive'])

def func_roc_auc_curve(model, valid_seq_x, valid_y):
    # generate a no skill prediction (majority class)
    ns_probs = [0 for _ in range(len(valid_y))]
    # predict probabilities
    lr_probs = model.predict_proba(valid_seq_x)
    # keep probabilities for the positive outcome only
    #lr_probs = lr_probs[:, 1]
    # calculate scores
    ns_auc = roc_auc_score(valid_y, ns_probs)
    lr_auc = roc_auc_score(valid_y, lr_probs)
    # summarize scores
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('NN: ROC AUC=%.3f' % (lr_auc))
    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(valid_y, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(valid_y, lr_probs)
    # plot the roc curve for the model
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(lr_fpr, lr_tpr, marker='.', label='NN')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the grid
    plt.grid(True)
    # show the legend
    plt.legend()
    # show the plot
    plt.show()
    
    
    
def func_precision_recall_curve(model, valid_seq_x, valid_y):
    # predict probabilities
    lr_probs = model.predict_proba(valid_seq_x)
    precision, recall, thresholds = precision_recall_curve(valid_y, lr_probs)
    # calculate precision-recall AUC
    lr_f1 = f1_score(valid_y, [1 if i>N else 0 for i in model.predict(valid_seq_x)]) 
    lr_auc = auc(recall, precision)
    # summarize scores
    print('Model: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
    # plot the precision-recall curves
    no_skill = len(valid_y[valid_y==1]) / len(valid_y)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    plt.plot(recall, precision, marker='.', label='NN')
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # show the legend
    plt.legend()
    # show the grid
    plt.grid(True)
    # show the plot
    plt.show()
    
    
    
def func_precision_recall(y_true, y_test, N=0.5,  verbose=True):
    r = [1 if i[1]>N else 0 for i in y_test]
    conf  = confusion_matrix(y_true, r)
    tn, fp, fn, tp = conf[0][0], conf[0][1], conf[1][0], conf[1][1]
    if verbose:
        print('''     Predicted       Predicted  
                 NO               YES
        Real   TN={}          FP={}
        NO     
        Real   FN={}           TP={}
        YES       '''.format(tn, fp, fn, tp))
        print('''
                  TP                     
    Precision = _______ = {}%    
                 TP+FP       


               TP
    Recall = ______  = {}%
              FN+TP           '''.format(round(tp/(tp+fp)*100,2), round(tp/(fn+tp)*100,2)))
    return round(tp/(tp+fp)*100,2), round(tp/(fn+tp)*100,2)

