import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder,StandardScaler
import pickle
import matplotlib.pyplot as plt


# def generate_scatter_plot(y_test, predicted_values):
#     fig, ax = plt.subplots()
#     ax.scatter(np.array(y_test), np.array(predicted_values))
#     ax.set_xlabel("Actual Values")
#     ax.set_ylabel("Predicted Values")
#     ax.set_title("Scatter Plot of Predicted Values")
#     return fig


data = pd.read_csv('Channel2.csv')
encoder = LabelEncoder()
scaler = StandardScaler()

X = data[['C/NO-1 (dB-Hz)', 'PR-1 (m)', 'Elevation-1 (deg)']]
y = data['output-1']
y = encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
y_true = np.array(y_test)

# ---------------Logistic regression model-------------------- #
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_predict_LR = log_reg.predict(X_test)
LR_accuracy = np.mean(y_predict_LR == y_test)
y_pred_LR = np.array(y_predict_LR)
cm_LR = confusion_matrix(y_true, y_pred_LR)
# fig = generate_scatter_plot(y_test, y_pred_LR)
# print(f"Logistic Regression Testing accuracy: {LR_accuracy}")

# # -----------------------KNN model---------------------------- #
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_predict_KNN = knn.predict(X_test)
KNN_accuracy = accuracy_score(y_test, y_predict_KNN)
y_pred_KNN = np.array(y_predict_KNN)
cm_KNN = confusion_matrix(y_true, y_pred_KNN)

# print(f"KNN Testing accuracy: {KNN_accuracy}")

# # --------------------SVM(linear kernel)---------------------- #
svm_linear = SVC(kernel='linear')
svm_linear.fit(X_train, y_train)
y_predict_svm_l = svm_linear.predict(X_test)
SVML_accuracy = accuracy_score(y_test, y_predict_svm_l)
y_pred_SVML = np.array(y_predict_svm_l)
cm_SVML = confusion_matrix(y_true, y_pred_SVML)

# print(f"SVM Linear Testing accuracy: {SVML_accuracy}")

# # --------------------SVM(rbf-kernel)------------------------- #
svm_rbf = SVC(kernel='rbf')
svm_rbf.fit(X_train, y_train)
y_predict_svm_r = svm_rbf.predict(X_test)
SVMr_accuracy = accuracy_score(y_test,y_predict_svm_r)
y_pred_SVMR = np.array(y_predict_svm_r)
cm_SVMR = confusion_matrix(y_true, y_pred_SVMR)

# print(f"SVM Kernel Testing accuracy: {SVMr_accuracy}")

# -------------------------testing----------------------------- #
# # 0 -> LOS , 1 -> MP
# # new_data = [[48.591255,36338610.35,59.087418]] # -> 1
# new_data = [[49.791569,36338639.21,59.087303]] # -> 0
# # new_data = scaler.fit_transform(new_data)
# LR_pred = log_reg.predict(new_data)
# print("output using Logistic Regression: ", LR_pred)

# KNN_pred = knn.predict(new_data)
# print("output using KNN model: ", KNN_pred)

# SVM_l = svm_linear.predict(new_data)
# print("output using SVM_l model: ", SVM_l)

# SVM_r = svm_rbf.predict(new_data)
# print("output using SVM_r model: ", SVM_r)

# -------------------------Pickle dumping----------------------------- #

pickle.dump(log_reg,open('model_lr.pkl','wb'))
model=pickle.load(open('model_lr.pkl','rb'))

pickle.dump(knn,open('model_knn.pkl','wb'))
model=pickle.load(open('model_knn.pkl','rb'))

pickle.dump(svm_linear,open('model_svml.pkl','wb'))
model=pickle.load(open('model_svml.pkl','rb'))

pickle.dump(svm_rbf,open('model_svmk.pkl','wb'))
model=pickle.load(open('model_svmk.pkl','rb'))