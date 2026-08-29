# Credit Card Fraud Detection: Experimental Protocol

**Protocol version:** - 1.0

**Status:** Final

**Project:** Credit Card Fraud Detection

**Repository:** 'NNikoletta/CreditCardFraudDetection'

## 1. Purpose

This file defines the experimental approach used to develop, compare, select, and evaluate credit card fraud detection models in this project. Its purpose is to ensure that experiments are reproducible, all the comparisons are fair, and the final test set remains isolated from model-development decisions.

All conducted experiments must follow this protocol.

## 2. Experimental principles

The following rules apply throughout this project:

1. Training, validation, and test data have distinct purposes and are isolated from each other.
2. The test set is only used for final evaluation and is not part of the model, hyperparameter, threshold selection process, early stopping, or debugging.
3. Every candidate model must be evaluated using the same saved data partition and the same predetermined trained seeds.
4. Preprocessing must be fitted only on the training data available withing the relevant experiment.
5. Every experimental choice must be supplied through configuration rather than by manually editing model or runner code.
6. Every run must create a new result code to ensure no existing experiment artifacts are being silently overwritten.
7. Results must be reported with enough metadata to identify the dataset, split, source-code version, configuration, and random seed that produces them.
8. A final test result may be used for reporting, but it must not initiate another round of model selection.
