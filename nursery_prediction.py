#This code predicts the ranking of a nursery school admission application into different classes. It is a multi-class classification problem.

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
from sklearn.metrics import roc_auc_score,mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from mlxtend.evaluate import bias_variance_decomp
from sklearn.ensemble import BaggingClassifier,RandomForestClassifier,VotingClassifier
from sklearn.preprocessing import LabelEncoder,LabelBinarizer
from sklearn.ensemble import IsolationForest
from imblearn.over_sampling import SMOTE
import category_encoders as ce
from  sklearn.metrics import precision_recall_fscore_support
from scipy import interp
from sklearn.metrics import roc_curve, auc



# Importing or Loading dataset
#data = "C:/Users/Gabriel/Desktop/nursery_data.csv"
df = pd.read_csv(data, delimiter=',')
df = df.iloc[0:6000]
# Inspect data
print('Nursery Data without imbalance and Outliers')
print('Inspect Data')
#print(df.head(20).to_string())
print('\n')
# Through inspection, the nursery dataset is a dataset of categorical type with no missing values but
# has imbalance multiple classes.

# Check data shape
print('Data Shape----------:',df.shape)
# Check Data Types
print('Check the Data Types----------')
print(df.info())
print('\n')


# Check missing values in the data
df3 = df.isnull().sum()
print('Missing values in each feature \n:-------------------------------')
print(df3) # NOTE: There are no missing values in the data

df['target'] = df['target'].replace(to_replace=['recommend','very_recom'],
                                    value='recommend')
print(df.head(20).to_string())

# Separate feature vectors from target labels
X = df.drop('target',axis=1)
#print(X.to_string())
y = df['target'].copy()

# Group data by class to see how the samples are distributed between the two classes
grp_data = df.groupby(y).size()
print(grp_data)

y.hist()
plt.title('Imbalanced Classes for Multi-class Classification')
plt.show()


# Transform categorical attributes to numeric attributes
feature_enc = ce.OneHotEncoder(handle_unknown='ignore', use_cat_names=True)
X = feature_enc.fit_transform(X)
#print(X.to_string())

# Check the descriptive statistics of each feature
for i in X:
    print(X[i].describe())
# As it can be seen from the statistical description of the data, the max value and std for some
# features are far from the mean given the median value. This shows the presence of outliers.

# Convert the Dataframe to Numpy Arrays
X = X.values
y = y.values
#print(y)

# Encoding attributes or Label Encoding: Transform the labels + to 0 and - to 1
enc = LabelEncoder()
y = enc.fit_transform(y)
print('Encoded Labels')
print(y)


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


# Visualize balanced classes
plt.hist(y_train)
plt.title('Balanced Class Distribution ')
plt.show()


# Performing feature normalization or standardization-----------------------------------
# The range of values for the attributes are of the same range


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
plt.savefig('Nursery_Hyperparameter_Output')
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

    # Classification Report: This gives us how often the algorithm predicted correctly
    def class_report(y_true, y_pred, y_score=None, average='micro'):
        if y_true.shape != y_pred.shape:
            print("Error! y_true %s is not the same shape as y_pred %s" % (
                y_true.shape,
                y_pred.shape)
                    )
            return

        lb = LabelBinarizer()

        if len(y_true.shape) == 1:
            lb.fit(y_true)

        # Value counts of predictions
        labels, cnt = np.unique(
            y_pred,
            return_counts=True)
        n_classes = len(labels)
        pred_cnt = pd.Series(cnt, index=labels)

        metrics_summary = precision_recall_fscore_support(
            y_true=y_true,
            y_pred=y_pred,
            labels=labels)

        avg = list(precision_recall_fscore_support(
            y_true=y_true,
            y_pred=y_pred,
            average='weighted'))

        metrics_sum_index = ['precision', 'recall', 'f1-score', 'support']
        class_report_df = pd.DataFrame(
            list(metrics_summary),
            index=metrics_sum_index,
            columns=labels)

        support = class_report_df.loc['support']
        total = support.sum()
        class_report_df['avg / total'] = avg[:-1] + [total]

        class_report_df = class_report_df.T
        class_report_df['pred'] = pred_cnt
        class_report_df['pred'].iloc[-1] = total

        if not (y_score is None):
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for label_it, label in enumerate(labels):
                fpr[label], tpr[label], _ = roc_curve(
                    (y_true == label).astype(int),
                    y_score[:, label_it])

                roc_auc[label] = auc(fpr[label], tpr[label])

            if average == 'micro':
                if n_classes <= 2:
                    fpr["avg / total"], tpr["avg / total"], _ = roc_curve(
                        lb.transform(y_true).ravel(),
                        y_score[:, 1].ravel())
                else:
                    fpr["avg / total"], tpr["avg / total"], _ = roc_curve(
                        lb.transform(y_true).ravel(),
                        y_score.ravel())

                roc_auc["avg / total"] = auc(
                    fpr["avg / total"],
                    tpr["avg / total"])

            elif average == 'macro':
                # First aggregate all false positive rates
                all_fpr = np.unique(np.concatenate([
                    fpr[i] for i in labels]
                ))

                # Then interpolate all ROC curves at this points
                mean_tpr = np.zeros_like(all_fpr)
                for i in labels:
                    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

                # Finally average it and compute AUC
                mean_tpr /= n_classes

                fpr["macro"] = all_fpr
                tpr["macro"] = mean_tpr

                roc_auc["avg / total"] = auc(fpr["macro"], tpr["macro"])

            class_report_df['AUC'] = pd.Series(roc_auc)

        return class_report_df
    clf_report = class_report(y_test, y_pred, y_score=model.predict_proba(X_test))
    print('Classification Report for', name, ':')
    print(clf_report)

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
