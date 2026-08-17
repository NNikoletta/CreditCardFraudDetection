Running regular cnn

epochs=5

batch=16



Params> 

Conv1d filters> 8, kernel> 29, stride>1, padding> same, activation>relu

Conv1d filters> 16, kernel> 29, stride>1, padding> same, activation>relu

Flatten

Dense nodes> 32

SoftMax out



**1st run**

Test loss:  0.005

Test accuracy: 99.93%

891/891 ━━━━━━━━━━━━━━━━━━━━ 2s 2ms/step

F1 score:  0.778

Recall score:  0.714

Precision score:  0.854

Average precision:  0.784

ROC-AUC:  0.949



&#x09;	CONFUSION MATRIX

LEGITIMATE (TRUE)	28426		6

FRAUD (TRUE)		14		35

&#x09;     		LEGITIMATE     FRAUD

&#x09;			PREDICTED





**2nd run**

Test loss:  0.004

Test accuracy: 99.937%

891/891 ━━━━━━━━━━━━━━━━━━━━ 2s 2ms/step

F1 score:  0.795

Recall score:  0.714

Precision score:  0.897

Average precision:  0.829

ROC-AUC:  0.951



&#x09;	CONFUSION MATRIX

LEGITIMATE (TRUE)	28428		4

FRAUD (TRUE)		14		35

&#x09;     		LEGITIMATE     FRAUD

&#x09;			PREDICTED





**3rd run**

Test loss:  0.005

Test accuracy: 99.93%

891/891 ━━━━━━━━━━━━━━━━━━━━ 2s 2ms/step

F1 score:  0.787

Recall score:  0.755

Precision score:  0.822

Average precision:  0.769

ROC-AUC:  0.945 



&#x09;	CONFUSION MATRIX

LEGITIMATE (TRUE)	28424		8

FRAUD (TRUE)		12		37

&#x09;     		LEGITIMATE     FRAUD

&#x09;			PREDICTED



**4th run**

Test loss:  0.004

Test accuracy: 99.937%

891/891 ━━━━━━━━━━━━━━━━━━━━ 2s 2ms/step

F1 score:  0.809

Recall score:  0.776

Precision score:  0.844

Average precision:  0.844

ROC-AUC:  0.948



&#x09;	CONFUSION MATRIX

LEGITIMATE (TRUE)	28428		7

FRAUD (TRUE)		11		38

&#x09;     		LEGITIMATE     FRAUD

&#x09;			PREDICTED



**5th run**

Test loss:  0.004

Test accuracy: 99.923%

891/891 ━━━━━━━━━━━━━━━━━━━━ 2s 2ms/step

F1 score:  0.784

Recall score:  0.816

Precision score:  0.755

Average precision:  0.85

ROC-AUC:  0.947



&#x09;	CONFUSION MATRIX

LEGITIMATE (TRUE)	28419		13

FRAUD (TRUE)		9		40

&#x09;     		LEGITIMATE     FRAUD

&#x09;			PREDICTED







AVERAGE RESULTS>

Test loss:  0.0044

Test accuracy: 99.9314%

F1 score:   0.7906 (std> 0.0107)

Recall score:  0.755 (std> 0.0388)

Precision score:  0.8344 (std> 0.0466)

Average precision:  0.8152

ROC-AUC:  0.948



OBSERVATION> Results seem to be a lot more stable, however runtime has increased considerably.

















