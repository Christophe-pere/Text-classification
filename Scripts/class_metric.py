import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import zero_one_loss
from sklearn.metrics import explained_variance_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
import seaborn as sns

class Metrics(object):
    '''
    Class containing functions to plot the different metric curves (Precision-Recall, ROC AUC etc...)
    '''
    
    def __init__(self):
        '''
        Initialisation of the class'''
        
        

    @classmethod 
    def func_roc_auc_curve(self, model, x, y, labels, gb=False):
        '''
        Function to plot the ROC AUC curves for binary or multiclass classification. 
        Correct for standard machine learning models and Neural Networks. 
        @param model: (model) classification model
        @param x: (list) validation sample
        @param y: (list int) validation sample label
        @param gb: (bool) inform if the model is an ensemble model 
        '''
        # generate a no skill prediction (majority class)
        ns_probs = [0 for _ in range(len(y.reshape(-1, 1)))]
        # predict probabilities
        if gb: # test if ensemble model 
            lr_probs = model.predict_proba(x)
        else:    
            lr_probs = model.predict(x)
        
        plt.figure(figsize=(10,8))
        if len(labels)==2: # binary classification 
            if gb:
                # compute area under the roc curve
                lr_auc = roc_auc_score(y, lr_probs[:,1], average="weighted")
            else:
                lr_auc = roc_auc_score(y, lr_probs, average="weighted")
            # Compute no skill roc curve 
            ns_fpr, ns_tpr, _ = roc_curve(y, ns_probs)
            # plot the curve no skill
            plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
            if gb: # test if the model is an ensemble model
                # compute ROC curve
                lr_fpr, lr_tpr, _ = roc_curve(y, lr_probs[:,1])
            else:
                # compute ROC curve
                lr_fpr, lr_tpr, _ = roc_curve(y, lr_probs)

            # plot the roc curve for the model
            plt.plot(lr_fpr, lr_tpr, label=f'Class (area {round(lr_auc,3)})')
                # axis labels
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
                # show the grid
            plt.grid(True)
                # show the legend
            plt.legend()
        else: # multilabel classification 
            dummy_y = np_utils.to_categorical(y)
            #ns_auc = roc_auc_score(valid_y, dummy_ns, average="weighted", multi_class="ovr")

            lr_auc_multi = []
            for i in enumerate(labels):
                lr_auc_multi.append(round(roc_auc_score(dummy_y[:,i[0]], lr_probs[:,i[0]], average="weighted"),3))
                print(f"ROC AUC class {i[1]}: {lr_auc_multi[-1]}")
            lr_auc = roc_auc_score(dummy_y, lr_probs, average="weighted", multi_class="ovr" )

            ns_fpr, ns_tpr = [i/10 for i in list(range(0, 11, 1))], [i/10 for i in list(range(0, 11, 1))] #f(range(0, 0.1, 1))
            plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
            for i in range(lr_probs.shape[1]):
                lr_fpr, lr_tpr, _ = roc_curve(dummy_y[:,i], lr_probs[:,i])
                # plot the roc curve for the model
                plt.plot(lr_fpr, lr_tpr, label=f'Class {labels[i]} (area {lr_auc_multi[i]})')
                # axis labels
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                # show the grid
                plt.grid(True)
                # show the legend
                plt.legend()
                
        print('\nROC AUC=%.3f \n' % (lr_auc))
        plt.show()

    @classmethod 
    def func_confusion_matrix(self, model, y, x, labels):
        '''
        Function to plot the ROC AUC curves for binary or multiclass classification. 
        Correct for standard machine learning models and Neural Networks. 
        @param model: (model) classification model
        @param x: (list) validation sample
        @param y: (list int) validation sample label
        
        '''
        if len(labels)==2: # binary confusion matrix
            confu_matrix = pd.DataFrame(confusion_matrix(y, (model.predict(x) > 0.5).astype(int)), \
                 columns=['Predicted Negative', "Predicted Positive"], index=['Actual Negative', 'Actual Positive'])
            print(confu_matrix)
            return confu_matrix
        else:
            # multiclass confusion matrix
            dummy_y = np_utils.to_categorical(y)
            mcm = multilabel_confusion_matrix(dummy_y, np_utils.to_categorical(model.predict(x).argmax(-1)))
            df_mcm = pd.DataFrame()
            for i in zip(mcm, labels): # compute confusion matrix for each class 
                mcm = pd.DataFrame(data=i[0], columns=['Predicted Negative', "Predicted Positive"], index=['Actual Negative', 'Actual Positive'])
                df_mcm = df_mcm.append(mcm)
                print("\nConfusion matrix for classe: %s \n" %(i[1]))
                print(mcm)
                print("\n")
            return df_mcm
        
    @classmethod 
    def func_precision_recall_curve(self, model, x, y, labels, gb=False):
        '''
        Function to plot the recall precision curves for binary or multiclass classification. 
        Correct for standard machine learning models and Neural Networks. 
        @param model: (model) classification model
        @param x: (list) validation sample
        @param y: (list int) validation sample label
        @param gb: (bool) inform if the model is an ensemble model 
        '''
        # predict probabilities
        if gb: # test if the model is an ensemble model 
            lr_probs = model.predict_proba(x)
        else:
            lr_probs = model.predict(x)

        print("\n")
        plt.figure(figsize=(10,8))

        if len(labels)==2: # binary classification 
            if gb:
                precision, recall, thresholds = precision_recall_curve(y, lr_probs[:,1]) # compute precision recall curve 
                lr_f1 = f1_score(y,(lr_probs[:,1]>0.5).astype(int))
            else:
                precision, recall, thresholds = precision_recall_curve(y, lr_probs)
                lr_f1 = f1_score(y,(lr_probs>0.5).astype(int))
            
            # calculate precision-recall AUC
            lr_auc = auc(recall, precision)
            # summarize scores
            print('Model: f1-score=%.3f AUC=%.3f' % (lr_f1, lr_auc)) # print f1-score and auc 
            plt.plot(recall, precision, marker='.', label='Model')
            no_skill = len(y[y==1]) / len(y)
            plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
        else:
            dummy_y = np_utils.to_categorical(y)
            dummy_lr = np_utils.to_categorical(lr_probs.argmax(-1))
            for i in enumerate(labels):
                precision, recall, thresholds = precision_recall_curve(dummy_y[:,i[0]], lr_probs[:,i[0]])
                # calculate precision-recall AUC
                lr_f1 = f1_score(dummy_y[:,i[0]], dummy_lr[:,i[0]]) 
                lr_auc = auc(recall, precision)
                # summarize scores
                print('Model class: %s --> f1-score=%.3f AUC=%.3f' % (i[1], lr_f1, lr_auc))
                plt.plot(recall, precision, label='Class %s' %(i[1]))
            no_skill = len(y[y>=1]) / len(y)
            plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
        # plot the precision-recall curves
        print("\n")

        # axis labels
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        # show the legend
        plt.legend()
        # show the grid
        plt.grid(True)
        # show the plot
        plt.show()

    @classmethod 
    def func_plot_eval_xgb(self, model, labels):
        '''
        Function to plot the evaluation curves for xgboost models 
        @param model: (model) xgboost model
        '''
        # retrieve performance metrics
        results = model.evals_result()
        if len(labels)>2: # multiclass 
            log_ = "mlogloss"
            error_= "merror"
        else: # binary classifiation
            log_ = "logloss"
            error_= "error"

        # create axis x with the number of epochs
        epochs = len(results['validation_0'][error_])
        x_axis = range(0, epochs)

        plt.figure(figsize=(15,10))
        plt.subplot(221)
        # Plot training & validation accuracy values
        plt.plot(x_axis, results['validation_0'][log_], label='Train')
        plt.plot(x_axis, results['validation_1'][log_], label='Test')
        plt.ylabel('Log Loss')
        plt.xlabel('Epochs')
        plt.title('XGBoost Log Loss')
        plt.legend(loc='upper left')
        plt.grid(True)


        # Plot training & validation loss values
        plt.subplot(222)
        plt.plot(x_axis, results['validation_0'][error_], label='Train')
        plt.plot(x_axis, results['validation_1'][error_], label='Test')
        plt.legend()
        plt.ylabel('Classification Error')
        plt.xlabel('Epochs')
        plt.title('XGBoost Classification Error')
        plt.legend( loc='upper left')
        plt.grid(True)
        plt.show()
        
    @classmethod 
    def plot_confusion_matrix(self, cm, classes, normalized=True, cmap='bone'):
        '''
        Function to generate an heatmap of the confusion matrix
        @param cm: (matrix) confusion matrix
        @param classes: (list) list containing labels of the classes
        @param normalised: (bool) determined if the confusion matrix need to be normalized
        @param cmap: (str) color for the confusion matrix
        '''
        plt.figure(figsize=[10, 8])
        norm_cm = cm
        if normalized:
            norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            sns.heatmap(norm_cm, annot=cm, fmt='g', xticklabels=classes, yticklabels=classes, cmap=cmap)
            #plt.savefig('confusion-matrix.png')

    @classmethod
    def func_plot_history(self, history):
        '''
        Function to plot the learning curves of a neural network
        @param history: metrics of a neural network
        '''
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
            
            
    @classmethod
    def func_metrics_deep_learning(self, model, history, x, y, labels):
        '''
        Function to plot the different metrics for the deep learning algorithms.
        @param model: (tensorflow.python.keras.engine.sequential.Sequential) deep learning model
        @param history: (tensorflow.python.keras.callbacks.History) history of the training model
        @param x: (numpy.ndarray) x data
        @param y: (numpy.ndarray) target data
        @param labels: (list) list containing the labels in str  
        '''
        self.func_plot_history(history)
        if len(labels)==2:

            print(classification_report(valid_y, (model.predict(x) > 0.5).astype(int), target_names=labels))
            print(f"\nThe balanced accuracy is : {round(100*balanced_accuracy_score(y, (model.predict(x)>0.5).astype(int)),2)}%\n")
            print(f"\nThe Zero-one Loss is : {round(100*zero_one_loss(y, (model.predict(x)>0.5).astype(int)),2)}%\n")
            print(f"\nExplained variance score: {round(explained_variance_score(y, (model.predict(x)>0.5).astype(int)),3)}\n" )
            self.func_roc_auc_curve(model, x, y, labels)
            self.func_precision_recall_curve(model, x, y, labels)

            print(f"\nCohen's kappa: {round(100*cohen_kappa_score(y, (model.predict(x) > 0.5).astype(int) ),2)}% \n") 
            matrices = self.func_confusion_matrix(model, y, x, labels)
            cm = confusion_matrix(y, (model.predict(x) > 0.5).astype(int))

            print("\nConfusion Matrix\n")
            self.plot_confusion_matrix(cm, labels)
        else:

            print(f"\nThe balanced accuracy is : {round(100*balanced_accuracy_score(y, model.predict(x).argmax(axis=-1)),2)}%\n")
            print(f"\nThe Zero-one Loss is : {round(100*zero_one_loss(y, model.predict(x).argmax(axis=-1)),2)}%\n")
            print(f"\nExplained variance score: {round(explained_variance_score(y, model.predict(x).argmax(axis=-1)),3)}\n" ) 
            self.func_roc_auc_curve(model, x, y, labels)
            self.func_precision_recall_curve(model, x, y, labels)

            print(f"\nCohen's kappa: {round(100*cohen_kappa_score(y, model.predict(x).argmax(axis=-1) ),2)}%\n")
            matrices = self.func_confusion_matrix(model, y, x, labels)
            cm = confusion_matrix(y, model.predict(x).argmax(axis=-1))

            print(classification_report(y, model.predict(x).argmax(axis=-1), target_names=labels))

            print("\nConfusion Matrix\n")
            self.plot_confusion_matrix(cm, labels)