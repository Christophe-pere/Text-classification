# Text-classification
This repository is made for text-classification. Different algorithms are implemented to class documents.
It contains machine learning implantation et deep learning implantation for binary et multiclass classification. 

This work has been presented in [Binary and Multiclass Text Classification (auto detection in a model test pipeline)](https://medium.com/@pere.christophe1/binary-and-multiclass-text-classification-auto-detection-in-a-model-test-pipeline-938158854943) waiting publication in [Towards Data Science](https://towardsdatascience.com/)


## Models 
### Machine Learning
The models implemented in the notebook for model selection are :
Multinomial Naive Bayes, logistic Regression, SVM, k-NN, Stochastic Gradient Descent, Gradient Boosting, XGBoost (Early stopping are implemented to stop the training to avoid overfitting).

### Deep Learning
The models implemented are:
Shallow Network, Deep Neaural Network, RNN, LSTM, CNN, GRU, CNN-LSTM, CNN-GRU, Bidirectional RNN, Bidirectional LSTM, Bidirectional GRU, RCNN (Early stopping are implemented to stop the training to avoid overfitting).

## Architecture of the notebook
> The class_metrics and CustomPreprocessing are available in the folder Scripts

- Module importation
- Parameters 
	- Here you'll choose the column name of the text to be classified and the name of the label column
	- Create Objects containing all the functions needed in the notebook
- List of Models
	- This variables are all boolean and permit to configure the type of models you want to test in the model selection
	- save_results is for the saving the finl dataframe containing the values of all metrics
	- lang is the parameter to detect the language of the data (API Google) if False, Engish is the default
	- sample is the parameter to choose a sample of the data (Default 5000 raws)
	- pre_trained is the parameter to use pretrained fastText model in the deep learning models
- Sand Box to Load Data
	- Here you will load your data and make manipulations on them to prepare them for the model selection 
- Start pipeline
	- If lang is True this part will detect the language of the text and select the most present in number of raws
- Polarity
	- Detect the polarity of the data with TextBlob
- Text informations
	- Compute the number of words, the number of character, the density
- Classes repartition
	- Show the quantity of text by label
- N-grams
	- Unigram
		- Show top words 
		- Show top words without stopwords
	- Bigrams
		- Show top words 
		- Show top words without stopwords
	- Trigrams
		- Show top words 
		- Show top words without stopwords
	- 5-grams
		- Show top words without stopwords
- Part of Speech
	- Extract Lemma, Pos and NER
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
	- Compute models and metrics
- All deep learning models
	- Compute models and metrics

--- 
Next steps:
- Use compressed layer with [TensorNet](https://github.com/google/TensorNetwork) like this [post](https://blog.tensorflow.org/2020/02/speeding-up-neural-networks-using-tensornetwork-in-keras.html) 
- Implement imbalanced methods to automaticaly balanced a dataset
- Use [Transformers](https://arxiv.org/abs/1706.03762) ([HuggingFace](https://huggingface.co/))
- Implement a Pre-trained transformers
- Test NLP with Reinforcement Learning
- Knowledge Graph
- Use distributed Deep Learning
- Use TensorNetwork to accelerate Neural Networks
- Select a class of models with the right method and does hyperparameters tuning 
- Use Quantum NLP (QNLP)
---
## Contribution
Your contributions are always welcome!

If you want to contribute to this list (please do), send me a pull request or contact me [@chris](twitter.com/Christo35427519) or [chris](linkedin.com/in/phdchristophepere)

--- 