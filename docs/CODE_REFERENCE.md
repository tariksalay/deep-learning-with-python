# Code and artifact reference

[Overview](../README.md) · [Walkthrough](PROJECT_GUIDE.md) · [Development guide](DEVELOPMENT.md)

This reference describes the files on this branch. Function names are navigation aids; the walkthrough explains how their inputs, state, and outputs fit together.

## Python modules

### [ICP-1.py](../ICP-1_Basics/ICP-1.py)

Console arithmetic and string/list practice.

Script-level exercise: read the dataset paths and execution statements before running it; there is no standalone function API.

### [ICP_2.py](../ICP-2_DataStructures/ICP_2.py)

Data-structure tasks, alternate-character extraction, and file-based word counting.

- `string_alternative()`

### [ICP_3.py](../ICP-3_WebScraping/ICP_3.py)

Employee inheritance examples plus HTML collection/parsing exercises.

- `Employee` — `__init__`, `avgSalary`
- `FullTimeEmployee`

### [ICP-4.py](../ICP-4_MLAlgorithms/ICP-4.py)

Glass-data classification with Gaussian Naive Bayes and SVM.

Script-level exercise: read the dataset paths and execution statements before running it; there is no standalone function API.

### [Preprocessing-EDA.py](../ICP-4_MLAlgorithms/Preprocessing-EDA.py)

Explore and prepare tabular train/test files for later classifiers.

Script-level exercise: read the dataset paths and execution statements before running it; there is no standalone function API.

### [knn.py](../ICP-4_MLAlgorithms/knn.py)

Fit and evaluate K-nearest neighbors on preprocessed tabular files.

Script-level exercise: read the dataset paths and execution statements before running it; there is no standalone function API.

### [svm.py](../ICP-4_MLAlgorithms/svm.py)

Fit and evaluate an SVM on preprocessed tabular files.

Script-level exercise: read the dataset paths and execution statements before running it; there is no standalone function API.

### [ICP_5.py](../ICP-5_RegressionModels/ICP_5.py)

Housing/wine regression experiments and exploratory plots.

Script-level exercise: read the dataset paths and execution statements before running it; there is no standalone function API.

### [preprocessing.py](../ICP-5_RegressionModels/preprocessing.py)

Explore the regression training table and prepare selected fields.

Script-level exercise: read the dataset paths and execution statements before running it; there is no standalone function API.

### [regression.py](../ICP-5_RegressionModels/regression.py)

Fit a regression exercise using the local training CSV.

Script-level exercise: read the dataset paths and execution statements before running it; there is no standalone function API.

### [Clusterusing.py](../ICP-6_KMeansClustering%2BPCA/Clusterusing.py)

K-means clustering on Iris measurements.

Script-level exercise: read the dataset paths and execution statements before running it; there is no standalone function API.

### [DataAnalysis.py](../ICP-6_KMeansClustering%2BPCA/DataAnalysis.py)

Iris preprocessing and principal-component exploration.

Script-level exercise: read the dataset paths and execution statements before running it; there is no standalone function API.

### [ICP_6.py](../ICP-6_KMeansClustering%2BPCA/ICP_6.py)

Credit-card clustering with scaling/PCA exploration.

Script-level exercise: read the dataset paths and execution statements before running it; there is no standalone function API.

### [PCA.py](../ICP-6_KMeansClustering%2BPCA/PCA.py)

Iris principal-component transformation.

Script-level exercise: read the dataset paths and execution statements before running it; there is no standalone function API.

### [ICP_7.py](../ICP-7_NLP/ICP_7.py)

NLTK processing and TF-IDF/classification exercises.

Script-level exercise: read the dataset paths and execution statements before running it; there is no standalone function API.

### [spelling_corrector.py](../ICP-7_NLP/spelling_corrector.py)

Generate edit-distance candidates and rank them by corpus frequency; preserve matched casing.

- `tokens(text)`
- `known(words)`
- `edits0(word)`
- `edits1(word)`
- `edits2(word)`
- `correct(word)`
- `correct_match(match)`
- `correct_text_generic(text)`

### [text_classification.py](../ICP-7_NLP/text_classification.py)

20 Newsgroups vectorization and Naive Bayes classification.

Script-level exercise: read the dataset paths and execution statements before running it; there is no standalone function API.

### [Airline_System.py](../LAB-1/Source/Airline_System.py)

Small OOP model of flights, people, employees, passengers, and tickets.

- `Flight` — `__init__`, `getInfo`
- `Person` — `__init__`
- `Passenger` — `__init__`, `getInfo`, `getFlightInfo`
- `Employee` — `__init__`, `getInfo`
- `Ticket` — `__init__`, `boardingInfo`

### [Inheritence.py](../LAB-1/Source/Inheritence.py)

Use the airline teaching classes in an inheritance exercise.

Script-level exercise: read the dataset paths and execution statements before running it; there is no standalone function API.

### [Question1.py](../LAB-1/Source/Question1.py)

Attempt a longest-substring-without-repetition exercise; edge cases are not comprehensively tested.

- `longestSubstring(string)`

### [Question2.py](../LAB-1/Source/Question2.py)

Script-level Python lab exercise; inspect prompts and data operations before execution.

Script-level exercise: read the dataset paths and execution statements before running it; there is no standalone function API.

### [Question4.py](../LAB-1/Source/Question4.py)

Collect/parse web content and write CSV output.

Script-level exercise: read the dataset paths and execution statements before running it; there is no standalone function API.

### [Question5.py](../LAB-1/Source/Question5.py)

Encode car attributes and compare Naive Bayes, KNN, and SVC.

Script-level exercise: read the dataset paths and execution statements before running it; there is no standalone function API.

### [Question6.py](../LAB-1/Source/Question6.py)

Cluster car attributes and inspect silhouette behavior.

Script-level exercise: read the dataset paths and execution statements before running it; there is no standalone function API.

### [Question7.py](../LAB-1/Source/Question7.py)

NLTK/WordNet and n-gram practice.

Script-level exercise: read the dataset paths and execution statements before running it; there is no standalone function API.

### [Question8.py](../LAB-1/Source/Question8.py)

Linear regression on candy attributes.

Script-level exercise: read the dataset paths and execution statements before running it; there is no standalone function API.

### [Q3.py](../LAB-2/Lab2/Lab2_Q3/Q3.py)

Load images from folders, train a CNN, save it, and make predictions.

- `read_data(path)`

### [Q4.py](../LAB-2/Lab2/Lab2_Q4/Q4.py)

CNN sentiment exercise using Phrase/Sentiment TSV columns.

Script-level exercise: read the dataset paths and execution statements before running it; there is no standalone function API.

### [Lab2_Q5.py](../LAB-2/Lab2/Lab2_Q5/Lab2_Q5.py)

LSTM sentiment exercise using Phrase/Sentiment TSV columns.

- `createmodel()`

### [Q7.py](../LAB-2/Lab2/Lab2_Q7/Q7.py)

MNIST autoencoder/reconstruction experiment.

Script-level exercise: read the dataset paths and execution statements before running it; there is no standalone function API.

### [q1.py](../LAB-2/Lab2/q1.py)

Dense-network regression using the diamonds table.

Script-level exercise: read the dataset paths and execution statements before running it; there is no standalone function API.

### [q2.py](../LAB-2/Lab2/q2.py)

Dense-network binary classification using heart data.

Script-level exercise: read the dataset paths and execution statements before running it; there is no standalone function API.

### [basicOP.py](../M2-ICP1_Keras/basicOP.py)

Keras dense networks for diabetes and breast-cancer tabular examples.

Script-level exercise: read the dataset paths and execution statements before running it; there is no standalone function API.

### [imageclassification.py](../M2-ICP2_NeuralNetworkFeatures/imageclassification.py)

MNIST dense-network classification.

Script-level exercise: read the dataset paths and execution statements before running it; there is no standalone function API.

### [sentiment_analysis.py](../M2-ICP3_ANN%2BRNN/sentiment_analysis.py)

Sentiment modeling from the local IMDB-related CSV.

Script-level exercise: read the dataset paths and execution statements before running it; there is no standalone function API.

### [image_classification.py](../M2-ICP4_ImageClassificationCNN/image_classification.py)

CIFAR-10 convolutional classification with an inverse-time learning-rate schedule.

Script-level exercise: read the dataset paths and execution statements before running it; there is no standalone function API.

### [sentimentAnalysis.py](../M2-ICP5_SentimentAnalysis%2BLSTM/sentimentAnalysis.py)

Embedding/LSTM classifier exposed to scikit-learn grid search via Keras SKLearnClassifier.

- `createmodel(X, y)`

### [autoencoder.py](../M2-ICP6_Autoencoder%2BPCA/autoencoder.py)

MNIST encoding and reconstruction experiment.

Script-level exercise: read the dataset paths and execution statements before running it; there is no standalone function API.

## Interpretation

A module without defined functions is still an executable experiment. Do not import or run this entire collection as a batch: dataset reads, prompts, downloads, plots, and training may occur at module level. `tests/test_compatibility.py` exercises selected compatibility paths, not every experiment or historical metric. See the walkthrough for dataset-schema differences and evaluation limits.
