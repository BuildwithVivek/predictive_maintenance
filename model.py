import numpy as np 
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import matplotlib.pyplot as plt
from sklearn import naive_bayes
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression


df = pd.read_csv('predictive_maintenance.csv')
df.set_index('UDI', inplace=True)
df.drop(['Product ID', 'Target'], axis=1, inplace=True)

le = LabelEncoder()
oe = OrdinalEncoder()

df['Type'] = le.fit_transform(df[['Type']])

X = df.drop(columns=['Failure Type'])
y = df['Failure Type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

tree_classifier = DecisionTreeClassifier(random_state=42,min_samples_split = 10, max_leaf_nodes = 15)
tree_classifier.fit(X_train, y_train)

naive_bayes = naive_bayes.GaussianNB()
naive_bayes.fit(X_train, y_train)

log_reg = LogisticRegression(penalty='l2', C=1.0)
log_reg.fit(X_train, y_train)

voting_clf = VotingClassifier(estimators=[
    ('logistic_regression', log_reg),
    ('decision_tree', tree_classifier),
    ('naive_bayes', naive_bayes)
], voting='hard')

voting_clf.fit(X_train, y_train)

joblib.dump(tree_classifier, 'decision_tree_model.joblib')
joblib.dump(naive_bayes, 'naive_bayes_model.joblib')
joblib.dump(log_reg, 'logistic_regression_model.joblib')

y_pred_dt = tree_classifier.predict(X_test)
y_pred_nb = naive_bayes.predict(X_test)
y_pred_lr = log_reg.predict(X_test)
y_pred_vc = voting_clf.predict(X_test)

accuracy_dt = accuracy_score(y_test, y_pred_dt)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
accuracy_vc = accuracy_score(y_test, y_pred_vc)

print("Accuracy:", accuracy_dt)
print("Accuracy:", accuracy_nb)
print("Accuracy:", accuracy_lr) 
print("Accuracy:", accuracy_vc)

plt.figure(figsize=(25, 10))  
plot_tree(tree_classifier, feature_names=X.columns, class_names=['0', '1', '2', '3', '4', '5'], filled=True, rounded=True)
plt.savefig('C:\\Users\\Aadya Dewangan\\Desktop\\pred_model\decision_tree_plot.png')  
