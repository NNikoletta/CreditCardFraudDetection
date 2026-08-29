# Credit Card Fraud Detection Project
  
This project focuses on a highly imbalanced credit card fraud detection dataset from Kaggle that contains credit card transactions made in September 2013. This is an end-to-end project utilizing machine learning methods, including deep learning as well as tree-based algorithms. The aim of this work is to compare the performance of various models and explore the possibility of using Convolutional Neural Networks for classification.

## Project Status

The project is in its finishing stages with the focus being on the refinement of the code.

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

It was also found that the data contains identical entries. These observations were collapsed into one representative row before data splitting to prevent repeated feature vectors from appearing across training, validation, and test partitions.

### Splitting the data

To ensure the reproducibility of the experiments, it is crucial to have a set of fixed splits of data. In total, six sets were created by saving the indices of each split: train, validate, and test. I used _scikit-learn’s_ _train_test_split_ method to create the indices, which were then saved in an _.npz_ format along with the split’s metadata in a _.json_ file. The first split was used for verification of the pipeline and later for final testing, while the following five splits marked as “experiment_split” were used in hyperparameter optimization and testing of the models. It is important to note, that all of the six splits have the same pool for the train-validate sets, while the test set remains unchanged in every split to avoid data leakage. Every train-validate split has a different random seed which is noted in the metadata. The data is split using an 80/10/10 ratio - train/validate/test respectively, except the final evaluation data where a 90/10 ratio is applied which corresponds to training/testing respectively. Since the validation is performed on the experimental data, the validation set can be removed during the final testing. This gives the model more entries to learn from which increases the classification accuracy. To address the imbalance, during the splitting of the datasets, stratification is used on the training and validation sets. To avoid accidentally mixing the data up, when loading the indices and applying them to the _x_ and _y_ arrays, two data types are created: DevelopmentData and TestData. The DevelopmentData contains the _x_train_, _x_validate, _y_train, and _y_validate arrays, while the TestData contains _x_test_ and _y_test_.

### Standardization of the data

Some tools that are utilized in this project, such as neural networks, tend to benefit from the data being centred around zero, that is why I decided to apply standardization. The _mean_ and _std_ of a few attributes were checked before proceeding with any modifications and the latter metric was varying. To fix this, _scikit-learn's_ _StandardScaler()_ is used after the data is separated into the _train_, _test_, and _validate_. The scaler is only fit on the training data to ensure that there is no data leakage and the performance of the model closely reflects the results it would show in a real life scenario.

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
The used MLP consists of three_Dense_ layers with the last one being the classification layer. The number of nodes is decreasing as we get deeper into the network. The architecture is deliberately kept small to see how a relatively lightweight model performs against Logistic Regression and later a CNN.
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
CNNConfig(batch_size=16, candidate_id=14, filters=(2, 4), kernel_size=(29, 15),
          strides=(1, 14), padding=('same', 'valid'), fc_units=16, threshold=0.3),
CNNConfig(batch_size=16, candidate_id=15, filters=(2, 4), kernel_size=(29, 29),
    	  strides=(1, 1), padding=('same', 'same'), fc_units=8, threshold=0.3),
CNNConfig(batch_size=16, candidate_id=16, filters=(4, 8), kernel_size=(29, 15),
    	  strides=(1, 14), padding=('same', 'same'), fc_units=8, threshold=0.3),
CNNConfig(batch_size=16, candidate_id=17, filters=(2, 4, 8), kernel_size=(29, 29, 29),
          strides=(1, 1, 1), padding=('same', 'same', 'valid'), fc_units=8),
CNNConfig(batch_size=16, candidate_id=18, filters=(2, 4, 8), kernel_size=(29, 15, 15),
          strides=(1, 1, 14), padding=('same', 'same', 'valid'), fc_units=8, threshold=0.3)
CNNConfig(batch_size=16, candidate_id=19, filters=(2, 4, 8), kernel_size=(29, 29, 29),
          strides=(1, 1, 1), padding=('same', 'same', 'valid'), fc_units=8, threshold=0.4)
CNNConfig(batch_size=16, candidate_id=20, filters=(2, 4, 8), kernel_size=(29, 29, 29),
          strides=(1, 1, 1), padding=('same', 'same', 'valid'), fc_units=8, threshold=0.45)
```

The experiments were focused on exploring how the number of extracted filters, kernel size, stride, padding, and number of fully connected units in the model affect its performance. While experimenting, multiple aspects were kept in mind, including the number of eventual trainable parameters, training time, and the recall-precision tradeoff. As I mentioned before, one can notice that the kernel size remains 29 in most of the candidates to see if any features can be extracted using a kernel that spans the whole frame. This is a purely experimental setup and the limitations it brings will be discussed in a later section of this work. As another experiment, the kernel size was set to 15 in candidates number 11, 13, 14, 16, and 18 which is roughly half of the whole data window. The reason for this was similar to the 29 size window. Because the position of the attributes due to their nature does not imply an immediate connection between them, I found the scientifically safest route to be a larger kernel size. Convolutional neural networks rely on a simple mathematical operation called convolution. This is where the network got its name from. Considering the proposed kernel size, and the type of operation used in the background (convolution), the number of filters is kept intentionally low, ranging from 2 to only 16. This is done to keep the background computations as low as possible, since it is one of the factors that affect the runtime. 

After experimenting with various parameters, based on the monitored training process and the validation results, another convolutional layer was added after the second one, making the total number of convolutional layers three. This decision was made because the model did not show substantial improvement during the tests.

The metrics I was focusing my model selection on were recall and precision, as well as the confusion matrices. My goal was to implement a model that creates a balance between the first two metrics. The main idea was to ensure that the number of false negatives and false positives is relatively balanced. If we assume that in a real life example a true positive is more costly to resolve than a false negative, or vice versa, it is better to have for instance, two FP and two FN than four of one or the other. This is a very simplified description of the system used for the model selection. To balance the recall and precision, I decided to lower the classification threshold, which decreases the precision but in return increases the recall. I continued keeping an eye on the confusion matrices as well to see how the balance between the FP and FN was maintained.

The averaged results of the experimental CNN runs are shown below:

| Model | Val Loss                 | Val Acc [%]                 | F1                       | Recall                   | Precision                | Avg precision            | ROC-AUC                  |
| ----- | ------------------------ | --------------------------- | ------------------------ | ------------------------ | ------------------------ | ------------------------ | ------------------------ |
| CNN1  | 0.00365<br>(std 0.00036) | 99.93253%<br>(std 0.00589%) | 0.79731<br>(std 0.01522) | 0.77872<br>(std 0.04381) | 0.82155<br>(std 0.04465) | 0.7951<br>(std 0.02034)  | 0.96948<br>(std 0.01151) |
| CNN2  | 0.00377<br>(std 0.00056) | 99.9289%<br>(std 0.01269%)  | 0.78734<br>(std 0.02476) | 0.76596<br>(std 0.03009) | 0.81624<br>(std 0.07066) | 0.80541<br>(std 0.01464) | 0.97002<br>(std 0.01426) |
| CNN3  | 0.00375<br>(std 0.00048) | 99.92454%<br>(std 0.0231%)  | 0.78707<br>(std 0.04386) | 0.79999<br>(std 0.02553) | 0.78402<br>(std 0.09778) | 0.74359<br>(std 0.05715) | 0.96857<br>(std 0.01213) |
| CNN4  | 0.00371<br>(std 0.00027) | 99.93325%<br>(std 0.01227%) | 0.80334<br>(std 0.02957) | 0.79574<br>(std 0.0217)  | 0.81303<br>(std 0.05215) | 0.76746<br>(std 0.05608) | 0.96699<br>(std 0.01122) |
| CNN5  | 0.00371<br>(std 0.00047) | 99.93253%<br>(std 0.01066%) | 0.80282<br>(std 0.01992) | 0.79999<br>(std 0.02553) | 0.81081<br>(std 0.06529) | 0.81683<br>(std 0.01609) | 0.96848<br>(std 0.01622) |
| CNN6  | 0.00447<br>(std 0.00181) | 99.90786%<br>(std 0.06124%) | 0.75558<br>(std 0.11476) | 0.76596<br>(std 0.03296) | 0.76177<br>(std 0.17228) | 0.74611<br>(std 0.12589) | 0.97256<br>(std 0.01415) |
| CNN7  | 0.00371<br>(std 0.00027) | 99.92963%<br>(std 0.01388%) | 0.79695<br>(std 0.03005) | 0.80426<br>(std 0.02481) | 0.79327<br>(std 0.06075) | 0.76746<br>(std 0.05608) | 0.96699<br>(std 0.01122) |
| CNN8  | 0.00371<br>(std 0.00047) | 99.91584%<br>(std 0.02785%) | 0.77109<br>(std 0.05439) | 0.80851<br>(std 0.02691) | 0.74812<br>(std 0.11191) | 0.81683<br>(std 0.01609) | 0.96848<br>(std 0.01622) |
| CNN9  | 0.00447<br>(std 0.00181) | 99.89625%<br>(std 0.06524%) | 0.74303<br>(std 0.10061) | 0.8<br>(std 0.01702)     | 0.71589<br>(std 0.16083) | 0.74611<br>(std 0.12589) | 0.97256<br>(std 0.01415) |
| CNN10 | 0.00383<br>(std 0.00046) | 99.92817%<br>(std 0.01244%) | 0.79538<br>(std 0.02722) | 0.81277<br>(std 0.02085) | 0.78273<br>(std 0.06628) | 0.81106<br>(std 0.02061) | 0.96952<br>(std 0.01719) |
| CNN11 | 0.00365<br>(std 0.0005)  | 99.9289%<br>(std 0.0168%)   | 0.79371<br>(std 0.03406) | 0.79149<br>(std 0.039)   | 0.8056<br>(std 0.09035)  | 0.80571<br>(std 0.02992) | 0.97248<br>(std 0.01326) |
| CNN12 | 0.00383<br>(std 0.00046) | 99.90786%<br>(std 0.03176%) | 0.75601<br>(std 0.05689) | 0.81277<br>(std 0.02085) | 0.71623<br>(std 0.10538) | 0.81106<br>(std 0.02061) | 0.96952<br>(std 0.01719) |
| CNN13 | 0.0037<br>(std 0.00025)  | 99.92164%<br>(std 0.03058%) | 0.77858<br>(std 0.05858) | 0.78298<br>(std 0.0434)  | 0.78763<br>(std 0.11304) | 0.79392<br>(std 0.03266) | 0.96706<br>(std 0.00877) |
| CNN14 | 0.00361<br>(std 0.00015) | 99.92672%<br>(std 0.01346%) | 0.78944<br>(std 0.03055) | 0.8<br>(std 0.01702)     | 0.78213<br>(std 0.06275) | 0.81571<br>(std 0.01624) | 0.97198<br>(std 0.01564) |
| CNN15 | 0.00405<br>(std 0.00071) | 99.91221%<br>(std 0.04245%) | 0.76723<br>(std 0.07499) | 0.80426<br>(std 0.01592) | 0.74866<br>(std 0.13458) | 0.80972<br>(std 0.03529) | 0.97581<br>(std 0.00801) |
| CNN16 | 0.00373<br>(std 0.00053) | 99.90568%<br>(std 0.05042%) | 0.75676<br>(std 0.0852)  | 0.80426<br>(std 0.01592) | 0.73261<br>(std 0.14718) | 0.80224<br>(std 0.0236)  | 0.97795<br>(std 0.0103)  |
| CNN17 | 0.00368<br>(std 0.0003)  | 99.93688%<br>(std 0.0148%)  | 0.81023<br>(std 0.0852)  | 0.78298<br>(std 0.03127) | 0.84508<br>(std 0.0745)  | 0.78904<br>(std 0.06153) | 0.97056<br>(std 0.00815) |
| CNN18 | 0.00395<br>(std 0.00044) | 99.91874%<br>(std 0.02208%) | 0.77357<br>(std 0.04418) | 0.8<br>(std 0.0217)      | 0.75566<br>(std 0.08903) | 0.79731<br>(std 0.03037) | 0.9717<br>(std 0.008)    |
| CNN19 | 0.00368<br>(std 0.0003)  | 99.92817%<br>(std 0.01365%) | 0.79208<br>(std 0.02909) | 0.79574<br>(std 0.0217)  | 0.79291<br>(std 0.06865) | 0.78904<br>(std 0.06153) | 0.97056<br>(std 0.00815) |
| CNN20 | 0.00368<br>(std 0.0003)  | 99.93035%<br>(std 0.01223%) | 0.79593<br>(std 0.02671) | 0.79149<br>(std 0.02481) | 0.80436<br>(std 0.06228) | 0.78904<br>(std 0.06153) | 0.97056<br>(std 0.00815) |

After examining all results, two models were chosen for final evaluation: CNN17 and CNN19. Candidate 17 shows higher results compared to all the other networks with the precision reaching 0.84508 and the overall accuracy and f1 being above 0.8. The reason CNN19 was also chosen is the balance between recall and precision. While both metrics are under 0.8, they are very close to each other and this was one of the main criteria. It is important to note that CNN17 and CNN19 are in reality the same model with only the threshold parameter being different. Lowering the threshold made the model more balanced. Although the two are the same, they are marked with different ids just to make sure it is easy to reference them. Evaluating them does not require retraining, if the model is trained, and the probabilities are saved, the results can be recreated by simply adjusting the threshold.


## Results

In the previous section, the focus was on choosing a CNN model based on the validation results. As mentioned before the chosen models were candidate number 17 and 19. In the results chapter of this work, the outputs of the CNN will be compared with the results produced by all models: Logistic Regression, MLP, and XGBoost. These outcomes can be categorized into two groups: results procured during the validation process, and results procured while running the final evaluation using the test data. Let’s first focus on the former group, the results of which can be found below:


| Model   | Val Loss                 | Val Acc [%]                 | F1                       | Recall                   | Precision                | Avg precision            | ROC-AUC                  |
| ------- | ------------------------ | --------------------------- | ------------------------ | ------------------------ | ------------------------ | ------------------------ | ------------------------ |
| LR      | 0.00455<br>(std 0.00043) | 99.91439%<br>(std 0.00846%) | 0.70544<br>(std 0.03652) | 0.60426<br>(std 0.04962) | 0.85042<br>(std 0.02364) | 0.72831<br>(std 0.0216)  | 0.97468<br>(std 0.00377) |
| MLP     | 0.00367<br>(std 0.00026) | 99.93253%<br>(std 0.00672%) | 0.79366<br>(std 0.02012) | 0.7617<br>(std 0.039)    | 0.83151<br>(std 0.03947) | 0.81835<br>(std 0.01441) | 0.96083<br>(std 0.01687) |
| CNN17   | 0.00368<br>(std 0.0003)  | 99.93688%<br>(std 0.0148%)  | 0.81023<br>(std 0.0852)  | 0.78298<br>(std 0.03127) | 0.84508<br>(std 0.0745)  | 0.78904<br>(std 0.06153) | 0.97056<br>(std 0.00815) |
| CNN19   | 0.00368<br>(std 0.0003)  | 99.92817%<br>(std 0.01365%) | 0.79208<br>(std 0.02909) | 0.79574<br>(std 0.0217)  | 0.79291<br>(std 0.06865) | 0.78904<br>(std 0.06153) | 0.97056<br>(std 0.00815) |
| XGBoost | 0.00276<br>(std 0.00016) | 99.95284%<br>(std 0.00513%) | 0.85038<br>(std 0.01758) | 0.78723<br>(std 0.03009) | 0.92589<br>(std 0.02362) | 0.83334<br>(std 0.01635) | 0.97732<br>(std 0.00572) |


During the final evaluation, the models were trained on the final_eval_split_v1 and the architectures were tested on the test dataset. The results were recorded and saved. Please see the outcomes below:

| Model   | Test Loss | Test Acc [%] | F1      | Recall  | Precision | Avg precision | ROC-AUC |
| ------- | -------- | ----------- | ------- | ------- | --------- | ------------- | ------- |
| LR      | 0.00447  | 99.92019%   | 0.7381  | 0.65957 | 0.83784   | 0.7177        | 0.97091 |
| MLP     | 0.00237  | 99.95284%   | 0.85057 | 0.78723 | 0.925     | 0.86435       | 0.99249 |
| CNN17   | 0.00286  | 99.9347%    | 0.80851 | 0.80851 | 0.80851   | 0.84648       | 0.97624 |
| CNN19   | 0.00286  | 99.9347%    | 0.80851 | 0.80851 | 0.80851   | 0.84648       | 0.97624 |
| XGBoost | 0.00254  | 99.95284%   | 0.85057 | 0.78723 | 0.925     | 0.86569       | 0.96911 |


As we can see, while the Convolutional Neural Network performed quite well during validation, if we focus on the final evaluation, it is clear that overall, its results are inferior to the MLP and XGBoost results. However, the CNN was able to achieve a recall-precision balance, which is its strength that can be used in the future. If the main objective is making sure the recall and precision are well balanced, the proposed CNN can be a reasonably good tool. The results of CNN17 and CNN19 are the same, which is completely understandable. The metrics are calculated from the test data, which means that when the results were evaluated on this specific data fraction, there were no probabilities in the range that the lower threshold affects.

The ranking of the models is the following: Logistic Regression < CNN < MLP < XGBoost.

This ranking shows that even with a higher kernel size, for this specific dataset, CNN is not able to achieve the same results as an MLP or XGBoost which produce nearly identical results. Deepening or widening the CNN may be a good approach to improve its performance, however, these modifications would inevitably lead to higher number of parameters as well as higher computational requirements which would increase runtime. It is also important to note, that in this work, all models except the CNN, are trained on their default values. I believe that optimizing their parameters may lead to better performance which might result in the CNN performance being completely outranked.

** Additional notes:**

Although the dataset is extremely imbalanced, one can notice that I am not modifying the class weights in any of my models. During my experiments, I ran multiple tests with balanced weights and with the default settings as well. After comparing the results, I decided to continue this project with the default weights. The reason behind this is the recall-precision trade-off. While introducing balancing techniques to make up for the imbalance in the data increased the recall quite substantially, the precision dropped which heavily affected the classification. Because the objective of this project is tied to a real-life problem, it is important to maintain the balance between the two metrics. Of course, this type of balance is always crucial, but in this scenario it becomes imperative since both cases - missing fraudulent transactions as well as classifying too many legitimate transactions as fraudulent - may bring forward expenses and inconvenience for both users and the company employing the fraud detection techniques.

One can also notice that there is an AttentionCNN present in the code, along with a results section for this model. This is an experimental architecture that attempts to create an attention layer that would enhance the attributes of interest to make it easier for the model to predict the right classes and focus on the most important data. After further consideration, I have decided to exclude this module from the project with a possibility of exploring it in a continuation of this work.

## Limitations

There are multiple limitations that need to be pointed out. First let’s focus on the limitations of the dataset. The project does not have access to the raw data. The data used in this work has already been processed through PCA which means that there is not a way to determine which attribute is connected to what original feature of the data. During PCA some critical features might have been completely or partially lost making the training process difficult.

The second limitation that needs to be addressed is one emerging from the architecture of the CNN itself. Because it is tailored to this specific PCA transformed dataset, it is a very rigid model. While it has not been investigated yet, it is likely that changing the order of the columns in the original dataset will affect the classification. In addition, the kernels were designed for this specific dataset as well, and would not achieve the same results with a different dataset. While it is possible to adjust the kernels to a different input, with bigger datasets the computational load caused by this would affect the performance negatively. The model is not robust enough.

The third observation is focused on the pipeline itself. While it is an end-to-end architecture, it is not yet able to process online data. The dataset is fully downloaded and split into training, validation, and testing sets, which means that the model is not built to process real-time incoming data.






