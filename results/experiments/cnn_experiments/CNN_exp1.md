Running regular cnn

epochs=5

batch=64



Params>

Conv1d filters> 8, kernel> 29, stride>1, padding> same, activation>relu

Conv1d filters> 16, kernel> 29, stride>1, padding> same, activation>relu

Flatten

Dense nodes> 32

SoftMax out



**1st run**

Test loss:  0.003

Test accuracy: 99.961%

891/891 ━━━━━━━━━━━━━━━━━━━━ 2s 3ms/step

F1 score:  0.887

Recall score:  0.878

Precision score:  0.896

Average precision:  0.889

ROC-AUC:  0.96



&#x09;	CONFUSION MATRIX

LEGITIMATE (TRUE)	28427		5

FRAUD (TRUE)		6		43

&#x09;     		LEGITIMATE     FRAUD

&#x09;			PREDICTED





**2nd run**

Test loss:  0.004

Test accuracy: 99.902%

891/891 ━━━━━━━━━━━━━━━━━━━━ 3s 3ms/step

F1 score:  0.611

Recall score:  0.449

Precision score:  0.957

Average precision:  0.82

ROC-AUC:  0.947



&#x09;	CONFUSION MATRIX

LEGITIMATE (TRUE)	28431		1

FRAUD (TRUE)		27		22

&#x09;     		LEGITIMATE     FRAUD

&#x09;			PREDICTED





**3rd run**

Test loss:  0.003

Test accuracy: 99.94%

891/891 ━━━━━━━━━━━━━━━━━━━━ 2s 2ms/step

F1 score:  0.817

Recall score:  0.776

Precision score:  0.864

Average precision:  0.864

ROC-AUC:  0.953



&#x09;	CONFUSION MATRIX

LEGITIMATE (TRUE)	28426		6

FRAUD (TRUE)		11		38

&#x09;     		LEGITIMATE     FRAUD

&#x09;			PREDICTED



**4th run**

Test loss:  0.003

Test accuracy: 99.926%

891/891 ━━━━━━━━━━━━━━━━━━━━ 2s 2ms/step

F1 score:  0.788

Recall score:  0.796

Precision score:  0.78

Average precision:  0.849

ROC-AUC:  0.964



&#x09;	CONFUSION MATRIX

LEGITIMATE (TRUE)	28421		11

FRAUD (TRUE)		10		39

&#x09;     		LEGITIMATE     FRAUD

&#x09;			PREDICTED



**5th run**

Test loss:  0.003

Test accuracy: 99.916%

891/891 ━━━━━━━━━━━━━━━━━━━━ 2s 2ms/step

F1 score:  0.676

Recall score:  0.51

Precision score:  1.0

Average precision:  0.86

ROC-AUC:  0.955



&#x09;	CONFUSION MATRIX

LEGITIMATE (TRUE)	28432		0

FRAUD (TRUE)		24		25

&#x09;     		LEGITIMATE     FRAUD

&#x09;			PREDICTED







AVERAGE RESULTS>

Test loss:  0.0032

Test accuracy: 99.929%

F1 score:  0.7558 (std> 0.0993)

Recall score:  0.6818 (std> 0.1698)

Precision score:  0.8994 (std> 0.0761)

Average precision:  0.8564

ROC-AUC:  0.9558



OBSERVATION> Standard deviation between runs is too high, unpredictable model.





















