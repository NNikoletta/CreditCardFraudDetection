Running regular cnn

epochs=5

batch=16



Params> 

Conv1d filters> 16, kernel> 29, stride>1, padding> same, activation>relu

Conv1d filters> 32, kernel> 29, stride>1, padding> same, activation>relu

Flatten

Dense nodes> 32

SoftMax out



**1st run**

Test loss:  0.004

Test accuracy: 99.926%

891/891 ━━━━━━━━━━━━━━━━━━━━ 2s 2ms/step

F1 score:  0.784

Recall score:  0.776

Precision score:  0.792

Average precision:  0.846

ROC-AUC:  0.956





&#x09;	CONFUSION MATRIX

LEGITIMATE (TRUE)	28422		10

FRAUD (TRUE)		11		38

&#x09;     		LEGITIMATE     FRAUD

&#x09;			PREDICTED





**2nd run**

Test loss:  0.005

Test accuracy: 99.933%

891/891 ━━━━━━━━━━━━━━━━━━━━ 2s 2ms/step

F1 score:  0.8

Recall score:  0.776

Precision score:  0.826

Average precision:  0.768

ROC-AUC:  0.914



&#x09;	CONFUSION MATRIX

LEGITIMATE (TRUE)	28424		8

FRAUD (TRUE)		11		38

&#x09;     		LEGITIMATE     FRAUD

&#x09;			PREDICTED





**3rd run**

Test loss:  0.005

Test accuracy: 99.94%

891/891 ━━━━━━━━━━━━━━━━━━━━ 2s 3ms/step

F1 score:  0.8

Recall score:  0.694

Precision score:  0.944

Average precision:  0.836

ROC-AUC:  0.96



&#x09;	CONFUSION MATRIX

LEGITIMATE (TRUE)	28430		2

FRAUD (TRUE)		15		34

&#x09;     		LEGITIMATE     FRAUD

&#x09;			PREDICTED



**4th run**

Test loss:  0.004

Test accuracy: 99.86%

891/891 ━━━━━━━━━━━━━━━━━━━━ 2s 2ms/step

F1 score:  0.31

Recall score:  0.184

Precision score:  1.0

Average precision:  0.846

ROC-AUC:  0.963



&#x09;	CONFUSION MATRIX

LEGITIMATE (TRUE)	28432		0

FRAUD (TRUE)		40		9

&#x09;     		LEGITIMATE     FRAUD

&#x09;			PREDICTED



**5th run**

Test loss:  0.005

Test accuracy: 99.895%

891/891 ━━━━━━━━━━━━━━━━━━━━ 2s 2ms/step

F1 score:  0.559

Recall score:  0.388

Precision score:  1.0

Average precision:  0.788

ROC-AUC:  0.955



&#x09;	CONFUSION MATRIX

LEGITIMATE (TRUE)	28432		0

FRAUD (TRUE)		30		19

&#x09;     		LEGITIMATE     FRAUD

&#x09;			PREDICTED







AVERAGE RESULTS>

Test loss:  0.0046

Test accuracy: 99.9108%

F1 score:   0.6506 (std> 0.1933)

Recall score:  0.5636 (std> 0.2376)

Precision score:  0.9177 (std> 0.0807)

Average precision:  0.8168

ROC-AUC:  0.9496



OBSERVATION> Runtime is way too long this way. After first run, results seem to be quite close to CNN\_exp3. But the process of training seems to be a little more consistent, with the validation loss slowly decreasing. Seems to be more stable so far. If attention block is added, the increase in the number of parameters might cause overfitting. Will try it out and will possibly add a Dropout layer.





FINAL THOUGHTS> Way to big of a gap. If the model is not performing well, the mistakes are HUGE, see run4 for reference and run5. This model will be excluded. No continuation.













