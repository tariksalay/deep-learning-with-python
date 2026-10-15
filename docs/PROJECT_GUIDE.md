# Python, Machine Learning, and Deep Learning Coursework: project walkthrough

[Overview](../README.md) · [Code reference](CODE_REFERENCE.md) · [Development guide](DEVELOPMENT.md)

## Navigate by topic

This is an exercise archive, not one installable model service. The [code reference](CODE_REFERENCE.md) maps every inspected Python source file to its purpose and defined objects.

| Folder | Main subject |
| --- | --- |
| [ICP-1_Basics](../ICP-1_Basics) | Console input and Python operations |
| [ICP-2_DataStructures](../ICP-2_DataStructures) | Lists, alternate characters, and file-based word counts |
| [ICP-3_WebScraping](../ICP-3_WebScraping) | Requests/BeautifulSoup and employee inheritance |
| [ICP-4_MLAlgorithms](../ICP-4_MLAlgorithms) | Gaussian Naive Bayes, SVM, KNN, and tabular preprocessing |
| [ICP-5_RegressionModels](../ICP-5_RegressionModels) | Regression and exploratory preprocessing |
| [ICP-6_KMeansClustering+PCA](../ICP-6_KMeansClustering%2BPCA) | K-means, scaling, and PCA |
| [ICP-7_NLP](../ICP-7_NLP) | Tokenization, spelling correction, and text classification |
| [M2-ICP1_Keras](../M2-ICP1_Keras) | Dense networks on tabular data |
| [M2-ICP2_NeuralNetworkFeatures](../M2-ICP2_NeuralNetworkFeatures) | MNIST classification and training plots |
| [M2-ICP3_ANN+RNN](../M2-ICP3_ANN%2BRNN) | IMDB-related sentiment experiments |
| [M2-ICP4_ImageClassificationCNN](../M2-ICP4_ImageClassificationCNN) | CIFAR-10 convolutional network |
| [M2-ICP5_SentimentAnalysis+LSTM](../M2-ICP5_SentimentAnalysis%2BLSTM) | Embeddings, LSTM, and grid search |
| [M2-ICP6_Autoencoder+PCA](../M2-ICP6_Autoencoder%2BPCA) | Autoencoding and reconstruction |
| [LAB-1](../LAB-1), [LAB-2](../LAB-2) | Additional Python/OOP and modeling exercises |

## Objects used inside the experiments

DataFrames carry tabular features and targets. Scikit-learn estimators store fitted classifiers, regressors, or clusters. Text tokenizers learn vocabulary mappings, and padded arrays carry token IDs. Keras models contain layers, optimizer/loss settings, and fitted weights. The spelling corrector instead uses a corpus-frequency counter and edit-candidate sets. The airline lab defines `Flight`, `Person`, `Passenger`, `Employee`, and `Ticket` as small teaching objects. These examples are independent; there is no shared airline/CNN/NLP pipeline.

## Three representative execution paths

**Classical classification:** `ICP-4_MLAlgorithms/ICP-4.py` reads adjacent `glass.csv`, separates `Type`, splits rows, fits Gaussian Naive Bayes and SVM, and prints held-out reports. Other scripts use different preprocessed training files and are not interchangeable entry points.

**Image classification:** the CIFAR-10 script loads Keras data, scales pixels, encodes labels, builds convolution/pooling/dense layers, and trains/evaluates. Its test builds the architecture and performs a synthetic forward pass without downloading the dataset or reproducing training.

**Sentiment grid search:** the LSTM exercise reads `Sentiment.csv`, removes neutral labels, tokenizes/pads text, and builds an embedding/LSTM model through `createmodel(X, y)`. Keras’ `SKLearnClassifier` connects that model to `GridSearchCV`; candidate `fit_kwargs` change batch size and epochs. The separate phrase labs use `Phrase`/`Sentiment`, while this CSV exercise uses `text`/`sentiment`. Both phrase preprocessing schemas now have a regression check.

## Side effects and evaluation boundaries

Most scripts remain executable experiments rather than import-safe library modules. The word-count exercise can append to `word.txt`; the spelling corrector reads `big.txt` at import time. Review source before importing or running it. Dataset licenses, external URLs, NLTK resources, and saved artifacts need their own checks.

Compatibility tests do not eliminate experimental-design problems. Some exercises fit preprocessing before splitting or use test data for validation. No new accuracy claim is made from those historical outputs. See [development](DEVELOPMENT.md) for requirements and the [original wiki](https://github.com/tariksalay/deep-learning-with-python/wiki) for course context.
