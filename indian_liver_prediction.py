The prediction of indian liver disease in a patient is provided in this code. It is a binary classification problem

import warnings
warnings.filterwarnings('ignore')

# Import Libraries
from numpy import mean,std
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB,BernoulliNB,MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold,cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score,mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from mlxtend.evaluate import bias_variance_decomp
from sklearn.ensemble import BaggingClassifier,RandomForestClassifier,VotingClassifier
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
#from sklearn.ensemble import IsolationForest
from imblearn.over_sampling import SMOTE
import category_encoders as ce

# Importing or Loading dataset
data = "C:/Users/Gabriel/Desktop/ILPD.csv"
features = ['Age','Gender','Total Bilirubin','Direct Bilirubin','Alkphos','Sgpt Alamine','Sgot Aspartate',
            'Total Protiens','Albumin','A/G Ratio', 'Class']

df = pd.read_csv(data, delimiter=',', names=features)
# Inspect data
print('Inspect Data')
#print(df.to_string())
# Through inspection, the indian liver dataset has numeric features except the gender attribute which is categorical.
# The dataset has missing values, suffers from outliers and class imbalance problem
# Check data shape
print('Data Shape----------:',df.shape)
# Check Data Types
print('Check the Data Types----------')
print(df.info())
print('\n')

# DATA PREPARATION--------------------------------------------------------
print('Data Preparation')
df['Gender'].hist()
plt.title('Gender Distribution')
plt.show()

# Check class labels in the data
df_class_labels = df['Class'].unique()
print(df_class_labels)

# Class label distribution among samples
class_dist = df['Class'].value_counts()
print(class_dist)

# Transform the Gender column from categorical variables to numeric values
gender_trans = ce.OneHotEncoder(handle_unknown='ignore')
df['Gender'] = gender_trans.fit_transform(df['Gender'])
print(df.to_string())

# Check missing values
df3 = df.isnull().sum()
print('Missing values in each feature \n:-------------------------------')
print(df3)

# Replace Missing Values: Delete columns or rows having missing values more than 30% or to imput values if less--------
df = df.replace('NaN', np.nan)
#print(df.to_string())
# fill missing values with mean column values since we are dealing with numeric values
df.fillna(df.mean(),inplace=True)
# count the number of nan values in each column
df_aft_rem = df.isnull().sum()
print('After Imputting: Missing values in each feature \n:-------------------------------')
print(df_aft_rem)
#print(df.to_string())

# Check feature relevance to the target through correlation matrix
plt.figure(figsize=(12,12))
sns.heatmap(df.corr(), annot=True)
plt.title('Correlation Matrix')
plt.show()


# Check the correlation against the pairplot to see the distribution of the data
#sns.pairplot(df, height=2.5)
#plt.show()


# Separate feature vectors from target labels
X = df.drop('Class',axis=1)
#print(X.to_string())
y = df['Class'].copy()
#print(y.to_string())

# Check statistical summary of the data
print('Statistical Summary of the Data------------')
print(X.describe()) # The spread of the data (standard deviation) is far from the mean for some features.
# Scaling or Normalization is needed


# Visualize the data using Histogram plots
# plot the histograms of all features or variable in the data
X.hist(sharex=False, sharey=False,  xlabelsize=1, ylabelsize=1, figsize=(6,6))
plt.show()
# Note: In the plots, The shape of the each graph can be Gaussian’, skewed or even has an exponential distribution.
# Density plots
X.plot(kind='density', subplots=True, layout=(5,5), sharex=False, legend=True, fontsize=1, figsize=(8,8))
plt.show()


# Convert the Dataframe to Numpy Arrays
X = X.values
y = y.values

# Encoding attributes or Label Encoding: Transform the labels 1 and 2 to binary 0 and 1 values
enc = LabelEncoder()
y = enc.fit_transform(y)
#print('Encoded Labels')
#print(y)


# Group data by class to see how the samples are distributed between the two classes
grp_data = df.groupby(y).size()
print('Liver patients:',grp_data[0])
print('Non-liver patients:',grp_data[1])


# Check Class Distribution for Imbalance: random undersampling, SMOTE or ensemble methods (Bagging, Boosting)
# Bagging and SMOTE are used for data resampling and to handle the class imbalance problem
# Visualize classes
#plt.hist(y)
#plt.title('Imbalanced Class Distribution ')
#plt.show()


print('\n')


# Performing feature normalization or standardization-----------------------------------
# The range of values for the attributes are of the same range
scale = MinMaxScaler()
X = scale.fit_transform(X)
#X_test = scale.transform(X_test)


#View Data after scaling
X_train = pd.DataFrame(X_train)
print('Scaled Training data')
print(X_train)

X_test = pd.DataFrame(X_test)
print('Scaled Test data')
print(X_test)


# Select Relevant features by evaluating feature importance (Dimensionality Reduction)---------------------------
# All features are relevant and used
# DATA PREPARATION ENDS HERE---------------------------------------------------------------------


# Split the dataset into the Training set and Test set-------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)


# Check Outliers-------------------------------------------
# identify outliers using Isolation Forest in the training dataset
# The data has been transformed between a range of 0 and 1. So there is no need to check for outliers
# identify outliers using Isolation Forest in the training dataset
iso = IsolationForest(contamination=0.1)
# Contamination argument is used to help estimate the number of outliers in the dataset.
# This is a value between 0.0 and 0.5 and by default is set to 0.1.
outl = iso.fit_predict(X_train)
# select all rows that are not outliers
remove_outl = outl != -1
X_train, y_train = X_train[remove_outl, :], y_train[remove_outl]
# summarize the shape of the updated training dataset
print('New data without outliers \n')
print(X_train.shape, y_train.shape)


# Handling Class Imbalance: Ensure that there are no synthetic data in the test data or validation data
# Synthetic data can only be in the training data
# Check Class Distribution for Imbalance: SMOTE and bagging ensemble methods are used to handle the class imbalance problem
smt = SMOTE(random_state=42)
X_train, y_train = smt.fit_resample(X_train, y_train)

# View the training data after oversampling
print('Viewing data after oversampling using resample SMOTE')
X_train = pd.DataFrame(X_train)
print(X_train.shape)

# Check the shape of the balanced feature vectors
print('New Feature vector shape:',X_train.shape)
print('New Class shape:',y_train.shape)

# Check Class Distribution
print('Class Distribution in Training Data')
print('Liver Patients:',sum(y_train==0))
print('Non-Liver Patients:',sum(y_train==1))



# Visualize balanced classes
plt.hist(y_train)
plt.title('Balanced Class Distribution ')
plt.show()
"""

print('\n')
# MODEL DEVELOPMENT BEGINS
print('# MODEL DEVELOPMENT BEGINS')
# Cross validation of 10 folds and 5 runs
cv_method = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=42)


# Hyperparameter Optimization

# get a voting ensemble of models
def NB_Ensemble():
    # Develop NB ensemble
    models = list()
    models.append(('NB1', GaussianNB(var_smoothing=1e-9)))
    models.append(('NB2', MultinomialNB(alpha=1.0)))
    models.append(('NB3', BernoulliNB(alpha=1.0)))
    models.append(('NB4', GaussianNB(var_smoothing=1e-5)))
    models.append(('NB5', MultinomialNB(alpha=0.5)))
    # define the voting ensemble
    NBE = VotingClassifier(estimators=models, voting='soft')
    return NBE

# define the base models
def kNN_Ensemble():
    # Develop kNN Ensemble
    models = list()
    models.append(('KNN1', KNeighborsClassifier(n_neighbors=1,p=2)))
    models.append(('kNN3', KNeighborsClassifier(n_neighbors=3, p=5)))
    models.append(('kNN5', KNeighborsClassifier(n_neighbors=5, p=2)))
    models.append(('kNN7', KNeighborsClassifier(n_neighbors=7, p=1)))
    models.append(('kNN9', KNeighborsClassifier(n_neighbors=9, p=5)))
    # define the voting ensemble
    kNNE = VotingClassifier(estimators=models, voting='soft')
    return kNNE

def DT_Ensemble():
    # Develop DT Ensemble
    models = list()
    models.append(('DT1', DecisionTreeClassifier(max_depth=5,criterion='entropy',splitter='best')))
    models.append(('DT2', DecisionTreeClassifier(max_depth=10,criterion='gini',splitter='best')))
    models.append(('DT3', DecisionTreeClassifier(max_depth=15,criterion='entropy',splitter='random')))
    models.append(('DT4', DecisionTreeClassifier(max_depth=20,criterion='gini',splitter='random')))
    models.append(('DT5', DecisionTreeClassifier(max_depth=25,criterion='gini',splitter='best')))
    # define the voting ensemble
    DTE = VotingClassifier(estimators=models,voting='soft')
    return DTE

Rand_Forest = RandomForestClassifier(n_estimators=10,criterion='gini',max_depth=None)

def SVM_Ensemble():
    # Develop SVM Ensemble
    models = list()
    models.append(('SVM1', SVC(probability=True, kernel='rbf', C=1.0,gamma=0.1)))
    models.append(('SVM2', SVC(probability=True, kernel='poly', C = 0.01, degree=3, gamma=0.01)))
    models.append(('SVM3', SVC(probability=True, kernel='sigmoid', C=0.5, gamma=0.001)))
    models.append(('SVM4', SVC(probability=True, kernel='rbf', C=0.1,gamma=1.0)))
    models.append(('SVM5', SVC(probability=True, kernel='poly', C = 0.25, degree=5, gamma=0.01)))
    # define the voting ensemble
    SVE = VotingClassifier(estimators=models, voting='soft')
    return SVE



def MLP_Ensemble():
    # Develop SVM Ensemble
    models = list()
    models.append(('MLP1', MLPClassifier(hidden_layer_sizes=(25,25,25),activation="relu",solver='adam',
                               learning_rate="adaptive",learning_rate_init=0.1, max_iter=1000)))
    models.append(('MLP2', MLPClassifier(hidden_layer_sizes=(50,25,25),activation="relu",solver='sgd',
                               learning_rate="constant",learning_rate_init=0.001, max_iter=1000)))
    models.append(('MLP3', MLPClassifier(hidden_layer_sizes=(50,25,50),activation="tanh",solver='lbfgs',
                               learning_rate="adaptive",learning_rate_init=0.0001, max_iter=1000)))
    models.append(('MLP4', MLPClassifier(hidden_layer_sizes=(50,50,50),activation="logistic",solver='sgd',
                               learning_rate="constant",learning_rate_init=0.01, max_iter=1000)))
    models.append(('MLP5', MLPClassifier(hidden_layer_sizes=(50,50,25),activation="tanh",solver='adam',
                               learning_rate="adaptive",learning_rate_init=0.00001, max_iter=1000)))
    # define the voting ensemble
    MLPE = VotingClassifier(estimators=models, voting='soft')
    return MLPE

print('\n')

# Developing heterogeneous ensemble
def get_HTRGN_ensemble():
    models = list()
    models.append(('NB_ensemble', NB_Ensemble()))
    models.append(('kNN_ensemble', kNN_Ensemble()))
    models.append(('DT_ensemble', DT_Ensemble()))
    models.append(('RF', Rand_Forest))
    models.append(('SVM_ensemble', SVM_Ensemble()))
    models.append(('MLP_ensemble', MLP_Ensemble()))
    HTE = VotingClassifier(estimators=models,voting='soft')
    return HTE

# Get a list of models to evaluate
def get_models():
    models = dict()
    models['NB_HE'] = NB_Ensemble()
    models['kNN_HE'] = kNN_Ensemble()
    models['DT_HE'] = DT_Ensemble()
    models['RF'] = Rand_Forest
    models['SVM_HE'] = SVM_Ensemble()
    models['ANN_HE'] = MLP_Ensemble()
    models['HTE'] = get_HTRGN_ensemble()
    return models


# evaluate a given model using cross-validation
def evaluate_model(model, X_train, y_train):
    scores = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=cv_method, n_jobs=-1)
    return scores

# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
print('Cross Validation Mean Accuracy and Std Dev of each Ensemble on test set:----------------------------------')
for name, model in models.items():
    scores = evaluate_model(model, X_test, y_test)
    results.append(scores)
    names.append(name)
    print('>%s %.3f' % (name, mean(scores)),u"\u00B1", '%.3f' % std(scores))

# plot model performance for comparison
plt.boxplot(results, labels=names, showfliers=False)
#plt.title('Cross validation Accuracy of ensembles')
#plt.title("Sonar_without_bag_{}".format(bagsize))
plt.xlabel("Ensembles")
plt.ylabel("Accuracy of Ensembles")
#plt.show()
plt.savefig('Indian_Liver_Hyperparameter_Output')
print('\n')

print('Cross Validation Mean Accuracy and Std Dev of each Ensemble on train set:-----------------------------')
for name, model in models.items():
    # evaluate the model
    scores = evaluate_model(model, X_train, y_train)
    # store the results
    results.append(scores)
    names.append(name)
    # summarize the performance along the way
    print('>%s %.3f' % (name, mean(scores)), u"\u00B1", '%.3f' % std(scores))

print('\n')
model_probab = list()
expert_prediction = list()
# Train and evaluate each Ensemble
for name,model in models.items():
    # fit the model
    model.fit(X_train,y_train)
    # then predict on the test set
    y_pred= model.predict(X_test)
    expert_prediction.append(y_pred)
    # Evaluate the models
    print('Performance Results of', name, ':----------------------------------------------------------')
    test_acc = accuracy_score(y_test,y_pred)
    y_pred1= model.predict(X_train)
    train_acc = accuracy_score(y_train,y_pred1)
    # Computing Generalizaton Factor
    test_err = 1-test_acc # generalization error
    train_err = 1-train_acc # training error
    gen_factor = test_err/train_err
    print('Accuracy and test error of', name, 'on test set:', test_acc,u"\u00B1",test_err)
    print('Actual label:',y_test)
    print('Predicted label:',y_pred)
    print('Accuracy and training error of', name, 'on train set:', train_acc,u"\u00B1",train_err)
    print('Generalization Factor to determine Ensemble Overfitting',gen_factor)
    # NOTE: if the gen_factor > 1, then the ensemble overfits else it is desirable
    # Classification Report: This gives us how often the algorithm predicted correctly
    clf_report= classification_report(y_test,y_pred)
    # Confusion Matrix: Showing the correctness and misclassifications made my the models
    conf = confusion_matrix(y_test, y_pred)
    print('Classification Report for', name,':')
    print(clf_report)
    print()
    print('Confusion Matrix for',name, ':')
    print(conf)
    print('\n')
    # Compute the probabilities of each ensemble to get ROC_AUC scores
    probs = model.predict_proba(X_test)
    model_probab.append(probs)
    # Evaluate Bias-Variance Tradeoff
    avg_expected_loss2, avg_bias2, avg_variance2 = bias_variance_decomp(model, X_train, y_train
                                                                            , X_test, y_test, loss='0-1_loss',
                                                                            num_rounds=10,
                                                                            random_seed=20)
    # Summary of Results
    print('Average Expected loss for', name, '%.2f' % avg_expected_loss2)
    print('Average Expected Bias error for', name, '%.2f' % avg_bias2)
    print('Average Expected Variance error for', name, '%.2f' % avg_variance2)
    print('\n')

# Obtain the probability scores of each ensemble
NB_HE_prob = model_probab[0]
kNN_HE_prob = model_probab[1]
DT_HE_prob = model_probab[2]
RF_prob = model_probab[3]
SVM_HE_prob = model_probab[4]
MLP_HE_prob = model_probab[5]
HTE_prob = model_probab[6]

# ROC_AUC Score
print('ROC AUC Score for each ensemble-------------------------------------------------------------')
model_auc1 = roc_auc_score(y_test, NB_HE_prob[:,1])
print('naive Bayes Ensemble: %.2f' % model_auc1)
model_auc2 = roc_auc_score(y_test, kNN_HE_prob[:,1])
print('kNN Ensemble: %.2f' % model_auc2)
model_auc3 = roc_auc_score(y_test, DT_HE_prob[:,1])
print('Decision Tree Ensemble: %.2f' % model_auc3)
model_auc4 = roc_auc_score(y_test, RF_prob[:,1])
print('Random Forest: %.2f' % model_auc4)
model_auc5 = roc_auc_score(y_test, SVM_HE_prob[:,1])
print('SVM Ensemble: %.2f' % model_auc5)
model_auc6 = roc_auc_score(y_test, MLP_HE_prob[:,1])
print('Neural Network Ensemble: %.2f' % model_auc6)
model_auc7 = roc_auc_score(y_test, HTE_prob[:,1])
print('Heterogeneous Ensemble: %.2f' % model_auc7)
print('\n')


print('Gathering Predictions of Experts----------------------------------------------------------')
# Expert Prediction
NB_HE_pred = expert_prediction[0]
kNN_HE_pred = expert_prediction[1]
DT_HE_pred = expert_prediction[2]
RF_pred = expert_prediction[3]
SVM_HE_pred = expert_prediction[4]
MLP_HE_pred = expert_prediction[5]
HTE_pred = expert_prediction[6]

# put each expert's predictions into a dataframe
df1 = pd.DataFrame(NB_HE_pred,columns=['NB_HE'])
df2 = pd.DataFrame(kNN_HE_pred,columns=['kNN_HE'])
df3 = pd.DataFrame(DT_HE_pred,columns=['DT_HE'])
df4 = pd.DataFrame(RF_pred,columns=['RF'])
df5 = pd.DataFrame(SVM_HE_pred,columns=['SVM_HE'])
df6 = pd.DataFrame(MLP_HE_pred,columns=['MLP_HE'])
df7 = pd.DataFrame(HTE_pred,columns=['HTE'])

# Put the dataframes into a list
df = [df1,df2,df3,df4,df5,df6,df7]
# Concatenate the dataframes
gather_pred = pd.concat(df,axis=1)
print(gather_pred)

