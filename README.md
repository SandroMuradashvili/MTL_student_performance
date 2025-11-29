Multi-Task Learning on Student Performance Dataset
1. Introduction

This project implements a two-headed MLP for predicting two outcomes from the UCI Student Performance Dataset:

G3: Final grade (regression)

Romantic: Whether the student is in a romantic relationship (classification)

The shared "body" learns a common representation, while each head specializes in its task.

2. Data Preprocessing

Categorical features were one-hot encoded.

Numerical features were standardized.

Data split into train, validation, and test sets.

3. Model Architecture

Shared Body: Two linear layers + ReLU + batch normalization + dropout

Head 1: Regression output for G3

Head 2: Classification output for romantic status

4. Training

Loss functions:

MSE for G3

CrossEntropyLoss for romantic

Weighted multi-task loss:

total_loss = alpha * loss_G3 + (1 - alpha) * loss_romantic


Optimizer: Adam, adjustable learning rate

Epochs: 30, Batch size: 32

5. Hyperparameter Exploration

We tried adjusting:

Model size (neurons in shared body and heads)

Data splits

Batch size and learning rate

Weighted loss coefficient alpha

Observation: Despite tuning, dataset features do not allow reliable prediction of romantic status. G3 prediction is robust.

6. Results
Effect of alpha
alpha	MAE (G3)	Accuracy (Romantic)	F1 (Romantic=Yes)
0.8	1.65	0.641	0.0667
0.5	2.07	0.615	0.0
0.2	1.88	0.602	0.1143
0.05	2.5	0.564	0.4333

Interpretation:

High alpha → prioritizes G3, suppresses romantic learning.

Very low alpha → improves F1 for romantic, small trade-off in G3.

Best observed F1: 0.4333 with alpha=0.05

7. Conclusion

Multi-task learning works well for G3 prediction.

Romantic status prediction is limited by dataset features.

Weighted loss with very low alpha balances the two tasks effectively.
