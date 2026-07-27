# Credit Card Fraud Detection Project
  
This project focuses on a highly imbalanced credit card fraud detection dataset from Kaggle that contains credit card transactions made in September 2013. This is an end-to-end project utilizing deep learning.

## A little bit about the data
### Data Source and Structure
The dataset contains credit card transactions made in Europe in September 2013. The dataset is extremely imbalanced with 492 fraudulent and 284,315 regular transactions. The project aims to create an end-to-end model that loads and processes the data before completing the detection which is followed by visualization of the choices the neural network made as well as the evaluation of the results.

The dataset is originally stored in a .csv file that consists of 31 attributes:
- Time - the time elapsed from the first transaction in seconds
- PCA Results (V1-V28)
- Amount - transaction value
- Class of the transaction:
  - 0: Legitimate transaction
  - 1: Fraudulent transaction
 
### Preparing the data
The dataset is first downloaded in a form of a .csv file. Before this however, the existence of the correct file path is checked. If the path exists, the pipeline checks the existence of the file, which is followed by data validation in case it already exists. This ensures that the data is not corrupted.

If the file is not found or it is corrupted, the code raises an error with an explanation that describes why we can not proceed with the processing of the data.

The data is then loaded into two numpy arrays _x_ and _y_. The former contains all the features while the latter stores all the classes that belong to each entry.

The _Time_ attribute is currently excluded from the data. This feature only represents the time elapsed from the very first transaction in seconds. Because the data source does not indicate what the exact relation is between the time of the experiment and the true time of the day, I am not able to reliably determine if a transaction was made during the day or night. Additionally, the data is not described as data coming from one subject only, which means that I am not able to separate any accounts to follow a shopping pattern where the _Time_ attribute would be valuable. Due to the reasons described above, I have decided to exlude the _Time_ column from this pipeline. Nevertheless, I am open to exploring the possibility of using this data in a future continuation of this project.


