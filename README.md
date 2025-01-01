# Credit Card Fraud Detection

## 1. Overview

Credit card fraud detection is a crucial task in the financial sector. Fraudulent transactions can result in massive financial losses and harm customer trust. Detecting fraud automatically can help prevent these issues by flagging suspicious activities.

- **Dataset**: 
   - The dataset contains historical credit card transactions.
   - It includes both fraudulent and non-fraudulent transactions.
   - Features in the dataset are anonymized to protect user privacy, with 30 features used for prediction, alongside the target variable (Class), where '1' indicates fraud and '0' indicates a legitimate transaction.
   - Dataset Distribution:
  <br>
<p align="center">
  <img src="https://raw.githubusercontent.com/leovidith/Credit-Card-Fraud-Detection/master/pie%20chart.png" width="300px">
</p>

- **Risks**:
   - **Imbalanced Data**: The dataset is highly imbalanced, with a very small percentage of fraudulent transactions, which can lead to biased predictions if not properly handled.
   - **False Negatives**: A misclassified fraudulent transaction as legitimate could result in substantial financial loss.
   - **Overfitting**: In some models, the risk of overfitting arises due to the imbalance or complex relationships in the data.

## 2. Results
Machine Learning Model Metrics:

| Model                      | Precision | Recall | F1-Score | Accuracy |
|----------------------------|-----------|--------|----------|----------|
| LogisticRegression          | 1.00      | 1.00   | 1.00     | 1.00     |
| DecisionTreeClassifier      | 1.00      | 1.00   | 1.00     | 1.00     |
| RandomForestClassifier      | 1.00      | 1.00   | 1.00     | 1.00     |
| KNeighborsClassifier        | 1.00      | 1.00   | 1.00     | 1.00     |
| SVC                         | 1.00      | 1.00   | 1.00     | 1.00     |
| MLPClassifier               | 1.00      | 1.00   | 1.00     | 1.00     |

Deep Neural Network Metrics:
1. Model Loss :  0.011761466972529888
2. Model Accuracy :  99.85%

<p align="center">
  <img src="https://github.com/leovidith/Credit-Card-Fraud-Detection/blob/master/cm.png" width="300px">
</p>

## 3. Agile Features

- **Sprints**: 
   - **Sprint 1**: Data preprocessing, feature engineering, and handling imbalanced data.
   - **Sprint 2**: Model development and training using Logistic Regression, Decision Trees, Random Forest, KNN, SVC, and MLP.
   - **Sprint 3**: Implementing and training the Deep Neural Network.
   - **Sprint 4**: Evaluation of model performance using classification metrics and improvement strategies.
   - **Sprint 5**: Model fine-tuning, testing, and final results presentation.

## 4. Conclusion

- **Summary**: 
   - The models performed exceptionally well in predicting fraudulent transactions, achieving near-perfect accuracy scores.
   - The **Deep Neural Network** also showed high performance with an accuracy of 99.84%, but the other traditional machine learning algorithms (Logistic Regression, Decision Trees, Random Forest, etc.) also showed similar results.
   - This indicates a robust model that can be used for credit card fraud detection, though balancing the dataset is crucial for optimizing the model performance in real-world scenarios.
