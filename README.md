# Credit Card Fraud Detection Project
  
This project focuses on a highly imbalanced credit card fraud detection dataset from Kaggle that contains credit card transactions made in September 2013. This is an end-to-end project utilizing machine learning methods, including deep learning as well as tree-based algorithms. The aim of this work is to compare the performance of various models and explore the possibility of using Convolutional Neural Networks for classification.

## Project Status

The project is in its experimental phase the focus of which is the optimization of hyperparameters.

## A little bit about the data
### Data Source and Structure
The [dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) contains credit card transactions made in Europe in September 2013. The dataset is extremely imbalanced with 492 fraudulent and 284,315 legitimate transactions (fraudulent transactions amount to ~0.172% of all data). The project aims to create an end-to-end model that automatically downloads, loads and processes the data before completing the classification task at hand which is the detection of fraudulent transactions. This is followed by the evaluation and visualization of the results.

The dataset is originally stored in a .csv file that consists of 31 attributes:
- Time - the time elapsed from the first transaction in seconds
- PCA Results (V1-V28)
- Amount - transaction value
- Class of the transaction:
  * 0: Legitimate transaction
  * 1: Fraudulent transaction
 
### Data Acquisition
The dataset is first downloaded as a .csv file. Before this however, the existence of the correct file path is checked. If the path exists, the pipeline checks the existence of the file, which is followed by data validation in case it already exists.

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

### Splitting the data

To ensure the reproducibility of the experiments, it is crucial to have a set of fixed splits of data. In total, six sets were created by saving the indices of each split: train, validate, and test. I used _scikit-learn’s_ _train_test_split_ method to create the indices, which were then saved in an _.npz_ format along with the split’s metadata in a _.json_ file. The first split was only used for verification of the pipeline, while the following five splits marked as “experiment_split” were used in hyperparameter optimization and testing of the models. It is important to note, that all of the six splits have the same pool for the train-validate sets, while the test set remains unchanged in every split to avoid data leakage. Every train-validate split has a different random seed which is noted in the metadata. The data is split using an 80/10/10 ratio - train/validate/test respectively. To address the imbalance, during the splitting of the datasets, stratification is used on the training and validation sets. To avoid accidentally mixing the data up, when loading the indices and applying them to the _x_ and _y_ arrays, two data types are created: DevelopmentData and TestData. The DevelopmentData contains the _x_train_, _x_validate, _y_train, and _y_validate arrays, while the TestData contains _x_test_ and _y_test_.

### Standardization of the data

Some tools that are utilized in this project, such as neural networks, tend to benefit from the data being centred around zero, that is why I decided to apply normalization. The _mean_ and _std_ of a few attributes were checked before proceeding with any modifications and the latter metric was varying. To fix this, _scikit-learn's_ _StandardScaler()_ is used after the data is separated into the _train_, _test_, and _validate_. The scaler is only fit on the training data to ensure that there is no data leakage and the performance of the model closely reflects the results it would show in a real life scenario.

On the other hand, some of the tools, such as XGBoost can simply use the unscaled data. Due to this another dataclass, ExperimentData, was created that is built using the previously mentioned DevelopmentData and TestData classes. It stores an unscaled and scaled version of the development data, as well as the test data. To make sure the fit scaler can later be used, this class also stores the scaler.

## Methods
Detection of fraudulent transactions is a difficult but necessary task. With credit cards becoming one of the main mediums of transactions in modern day-to-day life, it has become increasingly important to be able to accurately identify anomalies in shopping behaviour to avoid losses. I have decided to implement some architectures to compare various models and their performance on the previously described dataset. The following subchapters will describe the methods I have chosen. The two main families I focused on were neural networks and tree-based algorithms, I also explored linear classifiers such as Logistic Regression as a baseline for performance comparison.

### Linear methods - Logistic Regression

The Logistic Regression code can be found in the _linear.py_ file. By exploring a linear classifier first, I am able to see how utilizing more complex architectures affects the results of the classification.

While initially I was exploring various parameters, after a few experiments, I have decided to use the default configurations which are the following:
```
@dataclass(frozen=True)
class LogisticRegressionConfig:
    random_state: int = 42
    class_weight: str = None
    solver: str = 'lbfgs'
    penalty: str = 'l2'
    max_iter: int = 100
```

### Neural Networks
Neural networks are systems that have been inspired by the human brain’s ability to recognize patterns. Much like the brain, they were designed to be able to recognize underlying connections in a dataset.
I have explored various methods in an order of increased complexity. I used Logistic Regression as the baseline method for comparison, since it is one of the first classifiers that perform very well on linearly separable events. Next, I built a simple MLP (Multilayer Perceptron), which I followed with a CNN (Convolutional Neural Network).


#### Multilayer Perceptron (MLP)
The used MLP consists of three_Dense_ layers with the last one being the classification layer. The number of nodes is decreasing as we get deeper into the network. The architecture is deliberatly kept small to see how a relatively lightweight model performs against Logistic Regression and later a CNN.
```
input_main = Input(shape=(29,))

main_branch = Dense(32, activation=relu)(input_main)
main_branch = Dense(16, activation=relu)(main_branch)
sigmoid_out = Dense(1, activation=sigmoid)(main_branch)

```

#### Convolutional Neural Network (CNN)

In this project I am using Convolutional Neural Networks (CNNs) as an experimental method. Because CNNs were created for finding local patterns, they are mostly used to extract features from visual data like pictures as well as videos. Therefore, the benefit of using CNNs for credit card fraud detection needs to be experimentally proven.

The usage of CNNs would imply that there is a connection between the attributes located close to each other, however, for data like this, this hypothesis is methodologically flawed. Because the dataset contains results of PCA, a direct connection between two neighbouring columns cannot be assumed or proven. Therefore, in this project, I am using a kernel size of 29 which spans the whole attribute window to see if I am able to extract any features that would show a connection between all of the attributes.

Before the experiments are presented in a more detailed manner, it needs to be pointed out that reordering the attributes might affect the performance of the model due to the previously described nature of CNNs. Additionally, the usage of this method will inevitably make the model less robust, since it would be explicitly tied to the specific order of attributes it was trained on. The limitations will be addressed in a later section of this work.

The main architecture of the used CNN can be seen below:
```
input_main = Input(shape=(29,))
reshaped_main = Reshape((29, 1))(input_main)

main_branch = Conv1D(config.filters[0], kernel_size=config.kernel_size[0],
                     strides=config.strides[0], padding=config.padding[0],
                     activation=relu)(reshaped_main)
main_branch = Conv1D(config.filters[1], kernel_size=config.kernel_size[1],
                     strides=config.strides[1], padding=config.padding[1],
                     activation=relu)(main_branch)
main_branch = Flatten()(main_branch)

main_branch = Dense(self.config.fc_units, activation=relu)(main_branch)
sigmoid_out = Dense(1, activation=sigmoid)(main_branch)
```

Where _config_ is a an instance of the _CNNConfig_ dataclass with the following default values:
```
@dataclass(frozen=True)
class CNNConfig(TrainingConfig):
   candidate_id: int = 0
   filters: tuple[int, ...] = (8, 16)
   kernel_size: tuple[int, ...] = (29, 29)
   strides: tuple[int, ...] = (1, 1)
   padding: tuple[str, ...] = ('same', 'same')
   fc_units: int = 4

```

The CNN was run with various configurations on all five experimental data splits on the training and validation data for model selection, and the results were saved for future comparison.

### Decision-trees
As a comparison to deep learning methods, the project aims to explore decision-tree based models as well.

#### XGBoost
XGBoost is one of the most powerful algorithms that is widely used for credit card fraud detection. In this project, it is trained on unscaled data and the default parameters are used, which are the following:

```
@dataclass(frozen=True)
class XGBoostConfig:
   learning_rate: float = 0.1
   n_estimators: int = 100
   max_depth: int = 3
   random_state: int = 42
   gamma: float = 0
   min_child_weight: float = 1
   ratio: float = 1
```


## Experiments
Every run adheres to the [experimental protocol](https://github.com/NNikoletta/CreditCardFraudDetection/blob/main/docs/experimental_protocol.md). For model selection and initial performance comparison only the validation data was used and the models are trained on the five experimental data splits. Because the main experiments are focused on the optimization of the CNN, this section will focus on the process of tuning the hyperparameters. Every CNN has a _candidate_id_. In the following parts of this work, this is how the experiments will be referred to. The following setups were used for experimental purposes:

```
CNNConfig(batch_size=16, candidate_id=1, filters=(2, 4), kernel_size=(29, 29),
         strides=(1, 1), padding=('same', 'valid'), fc_units=4),
CNNConfig(batch_size=16, candidate_id=2, filters=(4, 8), kernel_size=(29, 29),
         strides=(1, 1), padding=('same', 'valid'), fc_units=4),
CNNConfig(batch_size=16, candidate_id=3, filters=(8, 16), kernel_size=(29, 29),
         strides=(1, 1), padding=('same', 'valid'), fc_units=4),
CNNConfig(batch_size=16, candidate_id=4, filters=(2, 4), kernel_size=(29, 29),
         strides=(1, 1), padding=('same', 'valid'), fc_units=8),
CNNConfig(batch_size=16, candidate_id=5, filters=(4, 8), kernel_size=(29, 29),
         strides=(1, 1), padding=('same', 'valid'), fc_units=8),
CNNConfig(batch_size=16, candidate_id=6, filters=(8, 16), kernel_size=(29, 29),
         strides=(1, 1), padding=('same', 'valid'), fc_units=8),
CNNConfig(batch_size=16, candidate_id=7, filters=(2, 4), kernel_size=(29, 29),
         strides=(1, 1), padding=('same', 'valid'), fc_units=8, threshold=0.3),
CNNConfig(batch_size=16, candidate_id=8, filters=(4, 8), kernel_size=(29, 29),
         strides=(1, 1), padding=('same', 'valid'), fc_units=8, threshold=0.3),
CNNConfig(batch_size=16, candidate_id=9, filters=(8, 16), kernel_size=(29, 29),
           strides=(1, 1), padding=('same', 'valid'), fc_units=8, threshold=0.3),
CNNConfig(batch_size=16, candidate_id=10, filters=(8, 16), kernel_size=(29, 29),
         strides=(1, 1), padding=('same', 'valid'), fc_units=16, threshold=0.3),
CNNConfig(batch_size=16, candidate_id=11, filters=(2, 4), kernel_size=(29, 15),
         strides=(1, 14), padding=('same', 'valid'), fc_units=8, threshold=0.3),
CNNConfig(batch_size=16, candidate_id=12, filters=(8, 16), kernel_size=(29, 29),
         strides=(1, 1), padding=('same', 'valid'), fc_units=16, threshold=0.2),
CNNConfig(batch_size=16, candidate_id=13, filters=(2, 4), kernel_size=(29, 15),
         strides=(1, 14), padding=('same', 'same'), fc_units=8, threshold=0.3),
CNNConfig(batch_size=16, candidate_id=13, filters=(2, 4), kernel_size=(29, 15),
         strides=(1, 14), padding=('same', 'valid'), fc_units=16, threshold=0.3)
```

The experiments were focused on exploring how the number of extracted filters, kernel size, stride, padding, and number of fully connected units in the model affect its performance. While experimenting, multiple aspects were kept in mind, including the number of eventual trainable parameters, training time, and the recall-precision tradeoff. As I mentioned before, one can notice that the kernel size remains 29 in most of the candidates to see if any features can be extracted using a kernel that spans the whole frame. This is a purely experimental setup and the limitations it brings will be discussed in a later section of this work. As another experiment, the kernel size was set to 15 in candidates number 11, 13, and 14, which is roughly half of the whole data window. The reason for this was similar to the 29 size window. Because the position of the attributes due to their nature does not imply an immediate connection between them, I found the scientifically safest route to be a larger kernel size. Convolutional neural networks rely on a simple mathematical operation called convolution. This is where the network got its name from. Considering the proposed kernel size, and the type of operation used in the background (convolution), the number of filters is kept intentionally low, ranging from 2 to only 16. This is done to keep the background computations as low as possible, since it is one of the factors that affect the runtime.

The metrics I was focusing my model selection on were recall and precision, as well as the confusion matrices. My goal was to create a model that creates a balance between the first two metrics. The main idea was to ensure that the number of false negatives and false positives is relatively balanced. If we assume that in a real life example the price of a true positive is more than the price of dealing with a false negative, or vice versa, it is better to have for instance two FP and two FN than four of one or the other. This is a very simplified description of the system used for the model selection. While running the experiments, I noticed that the precision is quite high in most of the cases, almost always reaching 0.9, and the recall is always falling behind in the 0.7-0.8 range. To make up for this, I decided to lower the classification threshold, which decreases the precision but in return increases the recall. I continued keeping an eye on the confusion matrices as well to see how the balance between the FP and FN was maintained.

The averaged results of the experimental CNN runs are shown below:
| Model | Val Loss                 | Val Acc [%]                  | F1                       | Recall                   | Precision                | Avg precision            | ROC-AUC                  |
| ----- | ------------------------ | ---------------------------- | ------------------------ | ------------------------ | ------------------------ | ------------------------ | ------------------------ |
| CNN1  | 0.00297<br>(std 0.00019) | 99.94944%<br>(std 0.00788%) | 0.83986<br>(std 0.02993) | 0.77551<br>(std 0.05772) | 0.9217<br>(std 0.04283)  | 0.83805<br>(std 0.00811) | 0.97574<br>(std 0.0149)  |
| CNN2  | 0.00302<br>(std 0.00018) | 99.94804%<br>(std 0.00716%)  | 0.83257<br>(std 0.0273)  | 0.7551<br>(std 0.04999)  | 0.93184<br>(std 0.03233) | 0.83483<br>(std 0.00742) | 0.96759<br>(std 0.01665) |
| CNN3  | 0.00328<br>(std 0.00047) | 99.94804%<br>(std 0.01097%)  | 0.83699<br>(std 0.03423) | 0.77551<br>(std 0.03872) | 0.91092<br>(std 0.04769) | 0.79424<br>(std 0.02721) | 0.96377<br>(std 0.01603) |
| CNN4  | 0.00298<br>(std 0.00023) | 99.94944%<br>(std 0.01008%)  | 0.84274<br>(std 0.03149) | 0.78776<br>(std 0.03558) | 0.90719<br>(std 0.04125) | 0.83187<br>(std 0.01306) | 0.97968<br>(std 0.01146) |
| CNN5  | 0.00304<br>(std 0.0003)  | 99.94874%<br>(std 0.00877%)  | 0.83502<br>(std 0.03258) | 0.75918<br>(std 0.05536) | 0.93182<br>(std 0.03227) | 0.83556<br>(std 0.01725) | 0.97802<br>(std 0.0104)  |
| CNN6  | 0.00318<br>(std 0.00034) | 99.94382%<br>(std 0.01422%)  | 0.81222<br>(std 0.06383) | 0.72653<br>(std 0.10212) | 0.93742<br>(std 0.04138) | 0.83911<br>(std 0.01073) | 0.97808<br>(std 0.0138)  |
| CNN7  | 0.00297<br>(std 0.00023) | 99.94523%<br>(std 0.00612%)  | 0.83516<br>(std 0.01298) | 0.80408<br>(std 0.02081) | 0.87153<br>(std 0.04799) | 0.83187<br>(std 0.01306) | 0.97968<br>(std 0.01146) |
| CNN8  | 0.00304<br>(std 0.0003)  | 99.94523%<br>(std 0.01187%)  | 0.82732<br>(std 0.04123) | 0.76735<br>(std 0.05714) | 0.89983<br>(std 0.03442) | 0.83556<br>(std 0.01725) | 0.97802<br>(std 0.0104)  |
| CNN9  | 0.00318<br>(std 0.00034) | 99.94804%<br>(std 0.00952%)  | 0.83852<br>(std 0.0288)  | 0.78367<br>(std 0.02768) | 0.90224<br>(std 0.03828) | 0.83911<br>(std 0.01073) | 0.97808<br>(std 0.0138)  |
| CNN10 | 0.00305<br>(std 0.00017) | 99.94804%<br>(std 0.00813%)  | 0.84161<br>(std 0.02114) | 0.8<br>(std 0.01527)     | 0.88922<br>(std 0.04401) | 0.83546<br>(std 0.01572) | 0.97534<br>(std 0.01015) |
| CNN11 | 0.00295<br>(std 0.00024) | 99.94944%<br>(std 0.01101%)  | 0.84456<br>(std 0.03171) | 0.79592<br>(std 0.02886) | 0.90144<br>(std 0.05393) | 0.84071<br>(std 0.00953) | 0.97012<br>(std 0.00885) |
| CNN12 | 0.00305<br>(std 0.00017) | 99.94663%<br>(std 0.00899%)  | 0.83876<br>(std 0.02374) | 0.80408<br>(std 0.01633) | 0.87798<br>(std 0.04625) | 0.83546<br>(std 0.01572) | 0.97534<br>(std 0.01015) |
| CNN13 | 0.00293<br>(std 0.00018) | 99.95295%<br>(std 0.00651%)  | 0.85446<br>(std 0.01608) | 0.8<br>(std 0.01527)     | 0.91918<br>(std 0.04895) | 0.83452<br>(std 0.01016) | 0.97157<br>(std 0.01068) |
| CNN14 | 0.00305<br>(std 0.00026) | 99.94452%<br>(std 0.00562%)  | 0.82972<br>(std 0.01967) | 0.78776<br>(std 0.04582) | 0.88058<br>(std 0.04024) | 0.83434<br>(std 0.01082) | 0.97675<br>(std 0.011)   |
| CNN15 | 0.00289<br>(std 0.00011) | 99.94733%<br>(std 0.00942%)  | 0.83767<br>(std 0.03036) | 0.79184<br>(std 0.04162) | 0.89057<br>(std 0.03143) | 0.84233<br>(std 0.00742) | 0.975<br>(std 0.01098)   |


## Results

In the previous section, the focus was on choosing a CNN model based on the validation results. As mentioned before the chosen model was candidate number -. In the results chapter of this work, the outputs of the CNN will be compared with the results produced by all models: Logistic Regression, MLP, and XGBoost. These outcomes can be categorized into two groups: results procured during the validation process, and results procured while running the final evaluation using the test data. Let’s first focus on the former group, the results of which can be found below:

Model	Val Loss	Val Acc [%]	F1	Recall	Precision	Avg precision	Roc/AUC


| Model        | Val Loss                 | Val Acc [%]                 | F1                       | Recall                   | Precision                | Avg precision            | Roc/AUC                  |
| ------------ | ------------------------ | --------------------------- | ------------------------ | ------------------------ | ------------------------ | ------------------------ | ------------------------ |
| LR           | 0.00405<br>(std 0.00059) | 99.91644%<br>(std 0.0134%)  | 0.71133<br>(std 0.04683) | 0.6<br>(std 0.05259)     | 0.87885<br>(std 0.06714) | 0.75958<br>(std 0.03442) | 0.97407<br>(std 0.01783) |
| MLP          | 0.00297<br>(std 0.00031) | 99.94452%<br>(std 0.01244%) | 0.8219<br>(std 0.03865)  | 0.74285<br>(std 0.03316) | 0.92059<br>(std 0.05297) | 0.84143<br>(std 0.0138)  | 0.96896<br>(std 0.01134) |
| CNN          |                          |                             |                          |                          |                          |                          |                          |
| AttentionCNN |                          |                             |                          |                          |                          |                          |                          |
| XGBoost      | 0.00263<br>(std 0.00026) | 99.95365%<br>(std 0.00871%) | 0.85158<br>(std 0.02962) | 0.77551<br>(std 0.04654) | 0.94779<br>(std 0.04303) | 0.8449<br>(std 0.01993)  | 0.97924<br>(std 0.01186) |



Although the dataset is extremely imbalanced, one can notice that I am not modifying the class weights in any of my models. During my experiments, I ran multiple tests with balanced weights and with the default settings as well. After comparing the results, I decided to continue this project with the default weights. The reason behind this is the recall-precision trade-off. While introducing balancing techniques to make up for the imbalance in the data increased the recall quite substantially, the precision dropped which heavily affected the classification. Because the objective of this project is tied to a real-life problem, it is important to maintain the balance between the two metrics. Of course, this type of balance is always crucial, but in this scenario it becomes imperative since both cases - missing fraudulent transactions as well as classifying too many legitimate transactions as fraudulent - may bring forward expenses and inconvenience for both users and the company employing the fraud detection techniques.






