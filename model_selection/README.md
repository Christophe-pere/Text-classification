# Model Selection for Text Classification

This notebook permit to make a selection model for text classification. The goal is to compare with different metrics machine learning and deep learning algorithms. It is configure to make classification with French and English texts.

## Models 
### Machine Learning
The models implemented in the notebook for model selection are :
Multinomial Naive Bayes, logistic Regression, SVM, k-NN, Stochastic Gradient Descent, Gradient Boosting, XGBoost (Early stopping are implemented to stop the training to avoid overfitting).

### Deep Learning
The models implemented are:
Shallow Network,Deep Neaural Network,RNN, LSTM, CNN, GRU, CNN-LSTM, CNN-GRU, Bidirectional RNN, Bidirectional LSTM, Bidirectional GRU, RCNN (Early stopping are implemented to stop the training to avoid overfitting).

## Architecture of the notebook
- Module importation
- Functions for metrics
- Parameters 
	- Here you'll choose the column name of the text to be classified and the name of the label column
- List of Models
	- This variables are all boolean and permit to configure the type of models you want to test in the model selection
	- save_results is for the saving the finl dataframe containing the values of all metrics
	- lang is the parameter to detect the language of the data (API Google) if False, Engish is the default
	- sample is the parameter to choose a sample of the data (Default 5000 raws)
	- pre_trained is the parameter to use pretrained fastText model in the deep learning models
- List of Metrics for the Model Selection
	- Contains the metrics considered for the model selection
	- They will be converted with make_scorer (sklearn) for the cross_validate function (sklearn)
- Sand Box to Load Data
	- Here you will load your data and make manipulations on them to prepare them for the model selection 
- Start pipeline
	- If lang is True this part will detect the language of the text and select the most present in number of raws
- Prepare data for ML Classic
	- Select a random sample of data (default 5000 raws) if sample is True
	- Select stopwords file in function of the language 
	- Create a new column for text without stopwords
- Class Weights 
	- Estimate the weight of each class present in the data and determine if the data is balanced or imbalanced
	- Work in progress, if the dataset is imbalanced create generic data with Smothe or Adasyn
- Machine learning
	- Save labels
	- Create empty dataframe to store the results of each metric for each model on each fold
	- Compute One-hot encoding
	- Compute TF-IDF
	- Compute TF-IDF n-grams (2, 3)
	- Compute TF-IDF n-grams characters (2, 3)
	- Load pretrained model fastText
	- Pad sentences in integers word vectors 
- All machine learning models
	- report () function based on cross_validate function to compute the metrics
- All deep learning models
	- cross_validate_NN() custom function for cross-validation (Stratified k-fold) and computed metrics
- Save the results if save_results if True

---
## Contribution
Your contributions are always welcome!

If you want to contribute to this list (please do), send me a pull request or contact me [@chris](twitter.com/Christo35427519) or [chris](linkedin.com/in/phdchristophepere)

--- 