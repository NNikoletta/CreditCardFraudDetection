Running regular cnn

epochs=5

batch=32



Params> 

Conv1d filters> 8, kernel> 29, stride>1, padding> same, activation>relu

Conv1d filters> 16, kernel> 29, stride>1, padding> same, activation>relu

Flatten

Dense nodes> 32

SoftMax out



**1st run**

Test loss:  0.004

Test accuracy: 99.937%

891/891 ━━━━━━━━━━━━━━━━━━━━ 2s 2ms/step

F1 score:  0.809

Recall score:  0.776

Precision score:  0.844

Average precision:  0.854

ROC-AUC:  0.947



&#x09;	CONFUSION MATRIX

LEGITIMATE (TRUE)	28425		7

FRAUD (TRUE)		11		38

&#x09;     		LEGITIMATE     FRAUD

&#x09;			PREDICTED





**2nd run**

Test loss:  0.004

Test accuracy: 99.937%

891/891 ━━━━━━━━━━━━━━━━━━━━ 2s 2ms/step

F1 score:  0.812

Recall score:  0.796

Precision score:  0.83

Average precision:  0.836

ROC-AUC:  0.961



&#x09;	CONFUSION MATRIX

LEGITIMATE (TRUE)	28424		8

FRAUD (TRUE)		10		39

&#x09;     		LEGITIMATE     FRAUD

&#x09;			PREDICTED





**3rd run**

Test loss:  0.003

Test accuracy: 99.902%

891/891 ━━━━━━━━━━━━━━━━━━━━ 2s 2ms/step

F1 score:  0.659

Recall score:  0.551

Precision score:  0.818

Average precision:  0.787

ROC-AUC:  0.971

&#x20; 



&#x09;	CONFUSION MATRIX

LEGITIMATE (TRUE)	28426		6

FRAUD (TRUE)		22		27

&#x09;     		LEGITIMATE     FRAUD

&#x09;			PREDICTED



**4th run**

Test loss:  0.003

Test accuracy: 99.951%

891/891 ━━━━━━━━━━━━━━━━━━━━ 2s 2ms/step

F1 score:  0.844

Recall score:  0.776

Precision score:  0.927

Average precision:  0.856

ROC-AUC:  0.951



&#x09;	CONFUSION MATRIX

LEGITIMATE (TRUE)	28429		3

FRAUD (TRUE)		11		38

&#x09;     		LEGITIMATE     FRAUD

&#x09;			PREDICTED



**5th run**

Test loss:  0.004

Test accuracy: 99.94%

891/891 ━━━━━━━━━━━━━━━━━━━━ 2s 2ms/step

F1 score:  0.817

Recall score:  0.776

Precision score:  0.864

Average precision:  0.833

ROC-AUC:  0.968



&#x09;	CONFUSION MATRIX

LEGITIMATE (TRUE)	28426		6

FRAUD (TRUE)		11		38

&#x09;     		LEGITIMATE     FRAUD

&#x09;			PREDICTED







AVERAGE RESULTS>

Test loss:  0.0036

Test accuracy: 99.9334%

F1 score:  0.7882 (std> 0.0658)

Recall score:  0.735 (std> 0.0923)

Precision score:  0.8566 (std> 0.0384)

Average precision:  0.8332

ROC-AUC:  0.9596



OBSERVATION> Better std, better overall results. Runtime not heavily affected but longer compared to batch=64 There is a definite precision trad off. The issue is however is in the recall. Recall is better with batch=32. So in general there is a precision/recall trade-off. BUT STD IS LOWER, MORE STABLE.



















