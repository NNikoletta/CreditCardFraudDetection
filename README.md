# Credit Card Fraud Detection Project
  
This project focuses on a highly imbalanced credit card fraud detection dataset from Kaggle that contains credit card transactions made in September 2013. This is an end-to-end project utilizing machine learning methods including deep learning as well as tree based algorithms.

## Project Status

Project is in its beginning stages with the current focus being on fine-tuning the existing code and researching the state-of-the-art methods.

## A little bit about the data
### Data Source and Structure
The dataset contains credit card transactions made in Europe in September 2013. The dataset is extremely imbalanced with 492 fraudulent and 284,315 legitimate transactions (fraudulent transactions amount to ~0.172% of all data). The project aims to create an end-to-end model that automatically downloads, loads and processes the data before completing the classification task at hand which is the detection of fraudulent transactions. This is followed by the evaluation and visualization of the results.

The dataset is originally stored in a .csv file that consists of 31 attributes:
- Time - the time elapsed from the first transaction in seconds
- PCA Results (V1-V28)
- Amount - transaction value
- Class of the transaction:
  - 0: Legitimate transaction
  - 1: Fraudulent transaction
 
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
8. Verifies if the number of fraudulent transactions is correct.
9. Verifies if the number of legitimate transactions is correct.
If the data passes every validation check, the code moves on to the next step.

The data is then loaded into two numpy arrays _x_ and _y_. The former contains all the features while the latter stores all the classes that belong to each entry.

The _Time_ attribute is currently excluded from the data. This feature only represents the time elapsed from the very first transaction in seconds. Because the data source does not indicate what the exact relation is between the time of the experiment and the true time of the day, I am not able to reliably determine if a transaction was made during the day or night. Additionally, the data is not described as data coming from one subject only, which means that I am not able to separate any accounts to follow a shopping pattern where the _Time_ attribute would be valuable. Due to the reasons described above, I have decided to exclude the _Time_ column from this pipeline. Nevertheless, I am open to exploring the possibility of using this data in a future continuation of this project.

### Normalization of the data

After the creation of the two previously mentioned numpy arrays that store the data (_x_, _y_), the focus falls on normalization. Because some of the tools in this project, such as neural networks, require the data to be centred around zero for the best possible performance, the data needs to be normalized. The _mean_ and _std_ of a few attributes were checked before proceeding with any modifications. Because the results of Principal Component Analysis (PCA) are very often already normalized, in certain cases there is no need for another normalization, however, in this scenario, the _std_ is varying. To fix this, _scikit-learn's_ _StandardScaler()_ is used after the data is separated into the _train_, _test_, and _validate_ sets in the ratio of ~80/10/10 respectively. The scaler is only fit on the training data to ensure that there is no data leakage and the performance of the model closely reflects the results it would show in a real life scenario.

On the other hand, the other used tools, such as _XGBoost_ do not require the data to be centred around zero. Due to this, the original data is also saved after the split for future use.

The labels are stored in one-dimensional arrays: _y_train_, _y_valid_, _y_test_.

Because this is a binary classification task, when neural networks are used, the model could use the one-dimensional data arrays as labels with the _sigmoid_ function at the classification step. However, I chose to add one-hot encoding for the labels. My choice is based purely on the possibility of later using and developing this code for a problem spanning over more classes.





**A brief summary of the data acquisition pipeline is described below:**
1. The data acquisition is completely automated by using Kaggle API. The _config.py_ inside the _src_ folder is responsible for handling all the private information to automatically access the dataset through the Kaggle username and key, which is stored in the _.env_ file. An _.env.example_ file is provided to give a clear picture of the file tree and flow of information.
2. The existence of the file path and the file itself is checked. The raw data is stored inside the project’s _data_ folder (./CreditCardFraudDetection/data/raw). If the folder does not exist, the code creates it.
3. The dataset is dowloaded automatically if it does not exist.
4. The validation of the data is performed.
5. The dataset is loaded into the previously mentioned numpy arrays.







