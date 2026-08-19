# Credit Card Fraud Detection Project
  
This project focuses on a highly imbalanced credit card fraud detection dataset from Kaggle that contains credit card transactions made in September 2013. This is an end-to-end project utilizing machine learning methods, including deep learning as well as tree-based algorithms. The aim of this work is to compare the performance of various models.

## Project Status

The project is in its experimental phase the focus of which is the optimization of hyperparameters.

## A little bit about the data
### Data Source and Structure
The dataset contains credit card transactions made in Europe in September 2013. The dataset is extremely imbalanced with 492 fraudulent and 284,315 legitimate transactions (fraudulent transactions amount to ~0.172% of all data). The project aims to create an end-to-end model that automatically downloads, loads and processes the data before completing the classification task at hand which is the detection of fraudulent transactions. This is followed by the evaluation and visualization of the results.

The dataset is originally stored in a .csv file that consists of 31 attributes:
- Time - the time elapsed from the first transaction in seconds
- PCA Results (V1-V28)
- Amount - transaction value
- Class of the transaction:
  * 0: Legitimate transaction
  * 1: Fraudulent transaction
 
### Data Acquisition
The dataset is first downloaded as a .csv file. Before this however, the existence of the correct file path is checked. If the path exists, the pipeline checks the existence of the file, which is followed by data validation in case it already exists. This ensures that the data is not corrupted.

If the file is not found, the file is downloaded and if it is corrupted, the code raises an error with an explanation that describes why we cannot proceed with the processing of the data.

The part responsible for validating the data is located inside the _src_ folder in the _validate_data.py_ file. The checks that the code performs are the following:
1. Verifies the existence of the file in the correct data path.
2. Verifies if the file contains anything at all by checking the size of the .csv.
3. Verifies if there are any data errors including any encoding or parsing errors.
4. Verifies if the number of columns is correct.
5. Verifies if the number of rows is correct.
6. Verifies if the column names match the expected column names.
7. Verifies if there are any empty cells (missing values).
8. Verifies if there are any unexpected labels in the class column.
9. Verifies if the number of fraudulent transactions is correct.
10. Verifies if the number of legitimate transactions is correct.
If the data passes every validation check, the code moves on to the next step.

The data is then loaded into two numpy arrays _x_ and _y_. The former contains all the features while the latter stores all the classes that belong to each entry.

The _Time_ attribute is currently excluded from the data. This feature only represents the time elapsed from the very first transaction in seconds. Because the data source does not indicate what the exact relation is between the time of the experiment and the true time of the day, I am not able to reliably determine if a transaction was made during the day or night. Additionally, the data is not described as data coming from one subject only, which means that I am not able to separate any accounts to follow a shopping pattern where the _Time_ attribute would be valuable. Due to the reasons described above, I have decided to exclude the _Time_ column from this pipeline. Nevertheless, I am open to exploring the possibility of using this data in a future continuation of this project.

### Normalization of the data

After the creation of the two previously mentioned numpy arrays that store the data (_x_, _y_), the focus falls on normalization. Some tools that are utilized in this project, such as neural networks, tend to benefit from the data being centred around zero, that is why I decided to apply normalization. The _mean_ and _std_ of a few attributes were checked before proceeding with any modifications. The results of Principle Component Analysis (PCA) often do not require any changes to be made, in certain cases there is no need for normalization, however, in this scenario, the _std_ is varying. To fix this, _scikit-learn's_ _StandardScaler()_ is used after the data is separated into the _train_, _test_, and _validate_ sets in the ratio of ~80/10/10 respectively.  To address the imbalance, during the splitting of the datasets, stratification is used on the training and validation sets. The scaler is only fit on the training data to ensure that there is no data leakage and the performance of the model closely reflects the results it would show in a real life scenario.

On the other hand, the other used tools, such as _XGBoost_ do not require the data to be centred around zero. Due to this, the original data is also saved after the split for future use.

The labels are stored in one-dimensional arrays: _y_train_, _y_valid_, _y_test_.

Because this is a binary classification task, when neural networks are used, the model could use the one-dimensional data arrays as labels with the _sigmoid_ function at the classification step. However, I chose to add one-hot encoding for the labels. My choice is based purely on the possibility of later using and developing this code for a problem spanning over more classes.


**Current data pipeline:**
1. The data acquisition is completely automated by using Kaggle API. The _config.py_ inside the _src_ folder is responsible for loading all the private information to automatically access the dataset through the Kaggle username and key, which is stored in the _.env_ file. An _.env.example_ file is provided to give a clear picture of the file tree and flow of information.
2. The existence of the file path and the file itself is checked. The raw data is stored inside the project’s _data_ folder (./CreditCardFraudDetection/data/raw). If the folder does not exist, the code creates it.
3. The dataset is downloaded automatically if it does not exist.
4. The validation of the data is performed.
5. The dataset is loaded into the previously mentioned numpy arrays.
6. Normalization is performed and both the original and the transformed datasets are stored for future use.

## Methods
Detection of fraudulent transactions is a difficult but necessary task. With credit cards becoming one of the main mediums of transactions in modern day-to-day life, it has become increasingly important to be able to accurately identify anomalies in shopping behaviour to avoid losses. After performing a review of state-of-the-art models as well as the way detectors developed over time, I have decided to implement some architectures to compare various models and their performance on the previously described dataset. The following subchapters will describe the methods I have chosen. The two main families I focused on were neural networks and tree-based algorithms, I also explored linear classifiers such as Logistic Regression as a baseline for performance comparison.

### Neural Networks
Neural networks are systems that have been inspired by the human brain’s ability to recognize patterns. Much like the brain, they were designed to be able to recognize underlying connections in a dataset.
The main builder blocks of a neural network are the nodes or neurons that are grouped into layers. Each node is connected to nodes in a following layer and the neurons are assigned weights the adjustment of which is the main objective of a process called training.
In the family of neural networks I have explored various methods in an order of increased complexity. I used Logistic Regression as the baseline method for comparison, since it is one of the first classifiers that perform very well on linearly separable events. Next, I built a simple MLP (Multilayer Perceptron), which I followed with a CNN (Convolutional Neural Network).
While Logistic Regression is not a neural network, since it consists of only one node that accepts inputs that are then weighted and sent to the sigmoid  function which outputs the probabilities, it is an excellent benchmark for future development. By exploring a linear classifier first, I am able to see how utilizing more complex architectures affects the results of the classification.

#### Logistic Regression
The Logistic Regression code can be found in the _neural_networks.py_ file, since I will be comparing its results with the performance of my MLP and CNN.
While initially I was exploring various parameters, after a few experiments, I have decided to use the default configurations which are the following:
```
@dataclass(frozen=True)
class LogisticRegressionConfig:
    random_state: int = 42
    class_weight: str = None  # 'balanced'
    solver: str = 'lbfgs'  # 'newton-cholesky'
    penalty: str = 'l2'
    max_iter: int = 100
```

#### Multilayer Perceptron (MLP)
The used MLP consists of four _Dense_ layers with the last one being the classification layer. The number of nodes is decreasing as we get deeper into the network. The architecture is deliberatly kept small to see how a relatively lightweight model performs against Logistic Regression and later a CNN.
```
input_main = Input(shape=(29,))

main_branch = Dense(64, activation=relu)(input_main)
main_branch = Dense(32, activation=relu)(main_branch)
main_branch = Dense(16, activation=relu)(main_branch)
softmax_out = Dense(2, activation=softmax)(main_branch)
```

#### Convolutional Neural Network (CNN)

**1. Regular CNN**
**2. Attention CNN**


## Experiments


Although, the dataset is extremely imbalanced, one can notice, that I am not modifying the class weights in any of my models. During my experiements, I ran multiple tests with balanced weights and with the defaults settings as well. After comparing the results, I decided to continue this project with the default weights. The reason behind this, is the recall-precision trade-off. While introducing balancing techniques to make up for the imbalance in the data increased the recall quite substantially, the precision dropped drastically which heavily affected the classification. Because the objective of this project is tied to a real-life problem, it is important to maintain the balance between the two metrics. Of course, this type of balance is always crucial, but in this scenario it becomes imperative since both cases - missing fraudulent transactions as well as classifying too many legitimate transactions as fraudulent - may bring forward expenses and inconvenience for both users and the company employing the fraud detection techniques.





# To-Do
- Create a data class for storing all the training, validation, and testing data.
- Explore Random Forrest for classification
- Explore Support Vector Machines







