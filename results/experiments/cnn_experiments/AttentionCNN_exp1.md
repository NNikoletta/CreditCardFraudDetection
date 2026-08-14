Running Attention CNN

epochs=5

batch=16



Params> 

Conv1d filters> 8, kernel> 29, stride>1, padding> same, activation>relu

AttributeAttention> kernel>2

Conv1d filters> 16, kernel> 29, stride>1, padding> same, activation>relu

Flatten

Dense nodes> 32

SoftMax out



**1st run**

Test loss:  0.003

Test accuracy: 99.933%

891/891 ━━━━━━━━━━━━━━━━━━━━ 2s 2ms/step

F1 score:  0.796

Recall score:  0.755

Precision score:  0.841

Average precision:  0.828

ROC-AUC:  0.943



&#x09;	CONFUSION MATRIX

LEGITIMATE (TRUE)	28425		7

FRAUD (TRUE)		12		37

&#x09;     		LEGITIMATE     FRAUD

&#x09;			PREDICTED





**2nd run**

Test loss:  0.004

Test accuracy: 99.93%

891/891 ━━━━━━━━━━━━━━━━━━━━ 2s 2ms/step

F1 score:  0.787

Recall score:  0.755

Precision score:  0.822

Average precision:  0.793

ROC-AUC:  0.944



&#x09;	CONFUSION MATRIX

LEGITIMATE (TRUE)	28424		8

FRAUD (TRUE)		12		37

&#x09;     		LEGITIMATE     FRAUD

&#x09;			PREDICTED





**3rd run**

Test loss:  0.004

Test accuracy: 99.926%

891/891 ━━━━━━━━━━━━━━━━━━━━ 2s 2ms/step

F1 score:  0.784

Recall score:  0.776

Precision score:  0.792

Average precision:  0.761

ROC-AUC:  0.952



&#x09;	CONFUSION MATRIX

LEGITIMATE (TRUE)	28422		10

FRAUD (TRUE)		11		38

&#x09;     		LEGITIMATE     FRAUD

&#x09;			PREDICTED



**4th run**

Test loss:  0.004

Test accuracy: 99.923%

891/891 ━━━━━━━━━━━━━━━━━━━━ 3s 3ms/step

F1 score:  0.766

Recall score:  0.735

Precision score:  0.8

Average precision:  0.805

ROC-AUC:  0.965



&#x09;	CONFUSION MATRIX

LEGITIMATE (TRUE)	28423		9

FRAUD (TRUE)		13		36

&#x09;     		LEGITIMATE     FRAUD

&#x09;			PREDICTED



**5th run**

Test loss:  0.004

Test accuracy: 99.923%

891/891 ━━━━━━━━━━━━━━━━━━━━ 2s 3ms/step

F1 score:  0.756

Recall score:  0.694

Precision score:  0.829

Average precision:  0.768

ROC-AUC:  0.958



&#x09;	CONFUSION MATRIX

LEGITIMATE (TRUE)	28425		7

FRAUD (TRUE)		15		34

&#x09;     		LEGITIMATE     FRAUD

&#x09;			PREDICTED







AVERAGE RESULTS>

Test loss:  0.0038

Test accuracy: 99.927%

F1 score:   0.7778 (std> 0.0146)

Recall score: 0.743 (std> 0.0277)

Precision score:  0.8042 (std> 0.0399)

Average precision: 0.791

ROC-AUC:  0.9524



OBSERVATION> Results are overall worse, than without the attention module. Try to move module to a different position.

















