# Spaceship Titanic Prediction Model

## Project Overview
This project implements an advanced machine learning solution for the Spaceship Titanic competition, predicting whether passengers were transported to an alternate dimension during the Spaceship Titanic's collision with a spacetime anomaly.

## Features
- Ensemble modeling with XGBoost, Random Forest, and LightGBM
- Advanced feature engineering
- Hyperparameter optimization
- Cross-validation
- Automated submission file generation

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Required Libraries
```bash
pip install pandas numpy scikit-learn xgboost lightgbm scipy
```

### Project Structure
```
spaceship-titanic/
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── src/
│   └── spaceship_model.py
│
├── submissions/
│   └── submission.csv
│
├── README.md
└── requirements.txt
```

## Usage

### Basic Usage
```python
from spaceship_model import create_submission

# Load data
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

# Create submission
submission_df, accuracy = create_submission(train_df, test_df)

# Save submission
submission_df.to_csv('submissions/submission.csv', index=False)
```

### Advanced Usage
```python
# With hyperparameter optimization
submission_df, accuracy = create_submission(
    train_df, 
    test_df, 
    use_optimization=True
)
```

## Feature Engineering

### Base Features
- Passenger Information
  - Age
  - VIP status
  - CryoSleep status
  - Home planet
  - Destination

### Engineered Features
1. **Spending Patterns**
   - Total spending across all services
   - Number of services used
   - Average spending per service
   - Spending variety index

2. **Cabin Analysis**
   - Deck extraction
   - Cabin number
   - Side information

3. **Group Features**
   - Group size based on last names
   - Family groupings

4. **Age-based Features**
   - Age categories (Child/Adult/Senior)
   - Age quantile binning

## Model Architecture

### Preprocessing Pipeline
```python
Pipeline([
    ('preprocessor', ColumnTransformer([
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(), categorical_cols),
        ('pass', 'passthrough', binary_cols)
    ])),
    ('classifier', VotingClassifier([
        ('xgb', XGBClassifier()),
        ('rf', RandomForestClassifier()),
        ('lgbm', LGBMClassifier())
    ]))
])
```

### Model Ensemble
- XGBoost Classifier
- Random Forest Classifier
- LightGBM Classifier
- Soft voting for final predictions

## Hyperparameter Optimization

### Parameters Tuned
- Learning rates
- Number of estimators
- Tree depths
- Sampling parameters

### Optimization Method
- RandomizedSearchCV with 5-fold cross-validation
- 50 iterations of parameter sampling
- Accuracy scoring metric

## Performance Metrics

### Validation Strategy
- 80/20 train-validation split
- 5-fold cross-validation
- Out-of-fold predictions

### Key Metrics
- Classification accuracy
- Cross-validation scores
- Parameter sensitivity

## Submission Format
```csv
PassengerId,Transported
0013_01,False
0018_01,False
0019_01,False
0021_01,False
```

## Future Improvements
1. Feature Selection
   - Implementation of LASSO/Ridge regularization
   - Feature importance analysis
   - Correlation analysis

2. Model Enhancements
   - Neural network integration
   - Stacking instead of voting
   - Bayesian optimization

3. Additional Features
   - Time-based features from passenger IDs
   - More sophisticated group features
   - Interaction terms

## Contributing
Feel free to submit issues and enhancement requests!

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Kaggle for providing the competition and dataset
- The scikit-learn, XGBoost, and LightGBM teams
- The Python data science community

## Contact
For questions and feedback, please open an issue in the repository.

---
Happy Predicting! 🚀✨
