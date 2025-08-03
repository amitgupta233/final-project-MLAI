# Data Sheet: Loan Default Dataset

## Source
- [Kaggle Lending Club Dataset](https://www.kaggle.com/datasets/wordsforthewise/lending-club)

## Structure
- 100,000 loan records
- 20+ features including demographic, loan, and credit history

## Preprocessing
- Handled missing values
- One-hot encoded categorical features
- Scaled numerical variables

## Data Access

This project uses the [Lending Club Loan Data](https://www.kaggle.com/datasets/wordsforthewise/lending-club). You can download the data directly with the script:

```python
import kagglehub
path = kagglehub.dataset_download("wordsforthewise/lending-club")

