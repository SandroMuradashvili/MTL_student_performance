# Multi-Task Learning: Student Performance Prediction

This project explores multi-task learning (MTL) with a two-headed MLP to predict **students' final grades (G3)** and **romantic status** from the UCI Student Performance Dataset.

## 1. Overview

The network handles **two tasks simultaneously**:

1. **Regression Task:** Predict the final grade (G3) of a student.  
2. **Classification Task:** Predict whether the student is in a romantic relationship (`yes`/`no`).

The network shares a **common feature extractor ("body")** and has **two separate heads** for each task.

## 2. Data Preprocessing

- Loaded `student-por.csv`.  
- Categorical features → one-hot encoding.  
- Numerical features → standardized.  
- Split dataset into **train, validation, test** (~75% / 13% / 12%).  
- Created **PyTorch Dataset and DataLoader**.

## 3. Model Architecture

- **Shared Body:** 2 fully-connected layers with BatchNorm, ReLU, Dropout.  
- **Grade Head (Regression):** 1 hidden layer → 1 output.  
- **Romantic Head (Classification):** 1 hidden layer → 2 output logits.  

## 4. Training

- Losses:
  - **MSELoss** for grades.  
  - **CrossEntropyLoss** for romantic status.  
- Weighted loss: `total_loss = alpha * grade_loss + (1 - alpha) * romantic_loss`.  
- Optimizer: **Adam** (lr=0.001).  
- Epochs: 30, batch size: 32.  

### Hyperparameter Experiments

- High `alpha` (~0.8, 0.5, 0.2):
  - MAE for grades good.  
  - F1 for romantic low (~0.06–0.17).  

- Low `alpha` (~0.05, 0.01):
  - MAE slightly worse (~2.5).  
  - F1 improved (~0.43).

## 5. Experiments

- Tested **network shapes**, **hidden sizes**, **dropout**, **batch size**, **learning rate**, and **loss weightings**.  
- **Observation:** Romantic status is hard to predict with available data. Best F1 ~0.4333 with `alpha=0.005`, MAE ~2.5.

## 6. Evaluation Metrics

- **Grades:** MAE  
- **Romantic status:** Accuracy, F1-score ("yes" class)

## 7. Example Results

| Alpha  | MAE (G3) | Accuracy (Romantic) | F1 (Yes) |
|--------|----------|-------------------|-----------|
| 0.8    | 1.6550   | 0.6410            | 0.0667    |
| 0.5    | 2.0723   | 0.6154            | 0.0000    |
| 0.2    | 1.8844   | 0.6026            | 0.1143    |
| 0.05   | 2.5000   | 0.5641            | 0.4333    |

## 8. Conclusion

- Multi-task learning predicts grades well.  
- Romantic status prediction remains difficult.  
- Small `alpha` values improve F1-score without majorly worsening grade MAE.

