# heart-disease-prediction
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
df = pd.read_csv('heart.csv')
df.head()

df.tail()

df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 303 entries, 0 to 302
Data columns (total 14 columns):

dtypes: float64(1), int64(13)
memory usage: 33.3 KB
df.describe()

df.isnull().sum()

dtype: int64
plt.figure(figsize=(22,10))

plt.xticks(size=20,color='grey')
plt.tick_params(size=12,color='grey')

plt.title('Null Value display',color='green',size=30)

sns.heatmap(df.isnull(),
            yticklabels=False,
            cbar=False,
            cmap='BuPu',
            )

import pandas_profiling as pp
pp.ProfileReport(df)

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
data = df.copy()
X = data.iloc[:,0:13]  
y = data.iloc[:,-1]    
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  
print(featureScores.nlargest(12,'Score'))  
 
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_) 
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(13).plot(kind='bar')
plt.show()

plt.figure(figsize=(12,10))
sns.heatmap(df.corr(),annot=True,cmap="BuGn",fmt='.2f')

for i in df.columns:
    print(i,len(df[i].unique()))

sns.set_style('darkgrid')
sns.set_palette('Set2')
df2 = df.copy()
def chng(sex):
    if sex == 0:
        return 'female'
    else:
        return 'male'
df2['sex'] = df2['sex'].apply(chng)
def chng2(prob):
    if prob == 0:
        return 'Heart Disease'
    else:
        return 'No Heart Disease'
df2['target'] = df2['target'].apply(chng2)
sns.countplot(data= df2, x='sex',hue='target')
plt.title('Gender versus target\n')
Text(0.5, 1.0, 'Gender versus target\n')

sns.countplot(data= df2, x='cp',hue='target')
plt.title('Chest Pain Type versus target\n')
Text(0.5, 1.0, 'Chest Pain Type versus target\n')

sns.countplot(data= df2, x='sex',hue='thal')
plt.title('Gender versus Thalassemia\n')
print('Thalassemia (thal-uh-SEE-me-uh) is an inherited blood disorder that causes your body to have less hemoglobin than normal. Hemoglobin enables red blood cells to carry oxygen')
Thalassemia (thal-uh-SEE-me-uh) is an inherited blood disorder that causes your body to have less hemoglobin than normal. Hemoglobin enables red blood cells to carry oxygen

sns.countplot(data= df2, x='slope',hue='target')
plt.title('Slope versus Target\n')
Text(0.5, 1.0, 'Slope versus Target\n')

sns.countplot(data= df2, x='exang',hue='thal')
plt.title('exang versus Thalassemia\n')
Text(0.5, 1.0, 'exang versus Thalassemia\n')

plt.figure(figsize=(16,7))
sns.distplot(df[df['target']==0]['age'],kde=False,bins=50)
plt.title('Age of Heart Diseased Patients\n')
C:\ProgramData\Anaconda2021\lib\site-packages\seaborn\distributions.py:2551: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
  warnings.warn(msg, FutureWarning)
Text(0.5, 1.0, 'Age of Heart Diseased Patients\n')

plt.figure(figsize=(16,7))
sns.distplot(df[df['target']==0]['chol'],kde=False,bins=40)
plt.title('Chol of Heart Diseased Patients\n')
C:\ProgramData\Anaconda2021\lib\site-packages\seaborn\distributions.py:2551: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
  warnings.warn(msg, FutureWarning)
Text(0.5, 1.0, 'Chol of Heart Diseased Patients\n')

plt.figure(figsize=(16,7))
sns.distplot(df[df['target']==0]['thalach'],kde=False,bins=40)
plt.title('thalach of Heart Diseased Patients\n')
C:\ProgramData\Anaconda2021\lib\site-packages\seaborn\distributions.py:2551: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
  warnings.warn(msg, FutureWarning)
Text(0.5, 1.0, 'thalach of Heart Diseased Patients\n')

df3 = df[df['target'] == 0 ][['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
       'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']] #target 0 - people with heart disease
pal = sns.light_palette("blue", as_cmap=True)


print('Age vs trestbps(Heart Diseased Patinets)')
sns.jointplot(data=df3,
              x='age',
              y='trestbps',
              kind='hex',
              cmap='Reds'
           
              )
Age vs trestbps(Heart Diseased Patinets)
<seaborn.axisgrid.JointGrid at 0x2237a0e4430>

sns.jointplot(data=df3,
              x='chol',
              y='age',
              kind='kde',
              cmap='PuBu'
              )
<seaborn.axisgrid.JointGrid at 0x22378c98ca0>

sns.jointplot(data=df3,
              x='chol',
              y='trestbps',
              kind='resid',
             
              )
<seaborn.axisgrid.JointGrid at 0x22378cd2c10>

sns.boxplot(data=df2,x='target',y='age')
<AxesSubplot:xlabel='target', ylabel='age'>

plt.figure(figsize=(14,8))
sns.violinplot(data=df2,x='ca',y='age',hue='target')
<AxesSubplot:xlabel='ca', ylabel='age'>

sns.boxplot(data=df2,x='cp',y='thalach',hue='target')
<AxesSubplot:xlabel='cp', ylabel='thalach'>

plt.figure(figsize=(10,7))
sns.boxplot(data=df2,x='fbs',y='trestbps',hue='target')
<AxesSubplot:xlabel='fbs', ylabel='trestbps'>

plt.figure(figsize=(10,7))
sns.violinplot(data=df2,x='exang',y='oldpeak',hue='target')
<AxesSubplot:xlabel='exang', ylabel='oldpeak'>

plt.figure(figsize=(10,7))
sns.boxplot(data=df2,x='slope',y='thalach',hue='target')
<AxesSubplot:xlabel='slope', ylabel='thalach'>

sns.violinplot(data=df2,x='thal',y='oldpeak',hue='target')
<AxesSubplot:xlabel='thal', ylabel='oldpeak'>

sns.violinplot(data=df2,x='target',y='thalach')
<AxesSubplot:xlabel='target', ylabel='thalach'>

sns.clustermap(df.corr(),annot=True)
<seaborn.matrix.ClusterGrid at 0x2237674dca0>

sns.pairplot(df,hue='cp')
<seaborn.axisgrid.PairGrid at 0x22374daf280>

from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split 
from sklearn import metrics 
X = df.iloc[:,0:13] 
y = df.iloc[:,13] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) 
clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
Accuracy: 0.7252747252747253
feature_cols = ['age', 'sex', 'cp', 'trestbps','chol', 'fbs', 'restecg', 'thalach','exang', 'oldpeak', 'slope', 'ca', 'thal']
import os

os.environ['PATH'] = os.environ['PATH']+';'+os.environ['CONDA_PREFIX']+r"\Library\bin\graphviz"

from sklearn.tree import export_graphviz
from six import StringIO  
from IPython.display import Image  
import pydotplus

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols  ,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('diabetes.png')
Image(graph.create_png())

clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
Accuracy: 0.7362637362637363
from six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('diabetes.png')
Image(graph.create_png())

df.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'rest_ecg_type', 'max_heart_rate_achieved',
       'exercise_induced_angina', 'st_depression', 'st_slope_type', 'num_major_vessels', 'thalassemia_type', 'target']

df.columns
Index(['age', 'sex', 'chest_pain_type', 'resting_blood_pressure',
       'cholesterol', 'fasting_blood_sugar', 'rest_ecg_type',
       'max_heart_rate_achieved', 'exercise_induced_angina', 'st_depression',
       'st_slope_type', 'num_major_vessels', 'thalassemia_type', 'target'],
      dtype='object')
df.loc[df['chest_pain_type'] == 0, 'chest_pain_type'] = 'asymptomatic'
df.loc[df['chest_pain_type'] == 1, 'chest_pain_type'] = 'atypical angina'
df.loc[df['chest_pain_type'] == 2, 'chest_pain_type'] = 'non-anginal pain'
df.loc[df['chest_pain_type'] == 3, 'chest_pain_type'] = 'typical angina'

df.loc[df['rest_ecg_type'] == 0, 'rest_ecg_type'] = 'left ventricular hypertrophy'
df.loc[df['rest_ecg_type'] == 1, 'rest_ecg_type'] = 'normal'
df.loc[df['rest_ecg_type'] == 2, 'rest_ecg_type'] = 'ST-T wave abnormality'

df.loc[df['st_slope_type'] == 0, 'st_slope_type'] = 'downsloping'
df.loc[df['st_slope_type'] == 1, 'st_slope_type'] = 'flat'
df.loc[df['st_slope_type'] == 2, 'st_slope_type'] = 'upsloping'

df.loc[df['thalassemia_type'] == 0, 'thalassemia_type'] = 'nothing'
df.loc[df['thalassemia_type'] == 1, 'thalassemia_type'] = 'fixed defect'
df.loc[df['thalassemia_type'] == 2, 'thalassemia_type'] = 'normal'
df.loc[df['thalassemia_type'] == 3, 'thalassemia_type'] = 'reversable defect'
df.head()
age	sex	chest_pain_type	resting_blood_pressure	cholesterol	fasting_blood_sugar	rest_ecg_type	max_heart_rate_achieved	exercise_induced_angina	st_depression	st_slope_type	num_major_vessels	thalassemia_type	target
0	63	1	typical angina	145	233	1	left ventricular hypertrophy	150	0	2.3	downsloping	0	fixed defect	1
1	37	1	non-anginal pain	130	250	0	normal	187	0	3.5	downsloping	0	normal	1
2	41	0	atypical angina	130	204	0	left ventricular hypertrophy	172	0	1.4	upsloping	0	normal	1
3	56	1	atypical angina	120	236	0	normal	178	0	0.8	upsloping	0	normal	1
4	57	0	asymptomatic	120	354	0	normal	163	1	0.6	upsloping	0	normal	1
data = pd.get_dummies(df, drop_first=False)
data.columns
Index(['age', 'sex', 'resting_blood_pressure', 'cholesterol',
       'fasting_blood_sugar', 'max_heart_rate_achieved',
       'exercise_induced_angina', 'st_depression', 'num_major_vessels',
       'target', 'chest_pain_type_asymptomatic',
       'chest_pain_type_atypical angina', 'chest_pain_type_non-anginal pain',
       'chest_pain_type_typical angina', 'rest_ecg_type_ST-T wave abnormality',
       'rest_ecg_type_left ventricular hypertrophy', 'rest_ecg_type_normal',
       'st_slope_type_downsloping', 'st_slope_type_flat',
       'st_slope_type_upsloping', 'thalassemia_type_fixed defect',
       'thalassemia_type_normal', 'thalassemia_type_nothing',
       'thalassemia_type_reversable defect'],
      dtype='object')
df_temp = data['thalassemia_type_fixed defect']
data = pd.get_dummies(df, drop_first=True)
data.head()
age	sex	resting_blood_pressure	cholesterol	fasting_blood_sugar	max_heart_rate_achieved	exercise_induced_angina	st_depression	num_major_vessels	target	chest_pain_type_atypical angina	chest_pain_type_non-anginal pain	chest_pain_type_typical angina	rest_ecg_type_left ventricular hypertrophy	rest_ecg_type_normal	st_slope_type_flat	st_slope_type_upsloping	thalassemia_type_normal	thalassemia_type_nothing	thalassemia_type_reversable defect
0	63	1	145	233	1	150	0	2.3	0	1	0	0	1	1	0	0	0	0	0	0
1	37	1	130	250	0	187	0	3.5	0	1	0	1	0	0	1	0	0	1	0	0
2	41	0	130	204	0	172	0	1.4	0	1	1	0	0	1	0	0	1	1	0	0
3	56	1	120	236	0	178	0	0.8	0	1	1	0	0	0	1	0	1	1	0	0
4	57	0	120	354	0	163	1	0.6	0	1	0	0	0	0	1	0	1	1	0	0
frames = [data, df_temp]
result = pd.concat(frames,axis=1)

result.head()
age	sex	resting_blood_pressure	cholesterol	fasting_blood_sugar	max_heart_rate_achieved	exercise_induced_angina	st_depression	num_major_vessels	target	...	chest_pain_type_non-anginal pain	chest_pain_type_typical angina	rest_ecg_type_left ventricular hypertrophy	rest_ecg_type_normal	st_slope_type_flat	st_slope_type_upsloping	thalassemia_type_normal	thalassemia_type_nothing	thalassemia_type_reversable defect	thalassemia_type_fixed defect
0	63	1	145	233	1	150	0	2.3	0	1	...	0	1	1	0	0	0	0	0	0	1
1	37	1	130	250	0	187	0	3.5	0	1	...	1	0	0	1	0	0	1	0	0	0
2	41	0	130	204	0	172	0	1.4	0	1	...	0	0	1	0	0	1	1	0	0	0
3	56	1	120	236	0	178	0	0.8	0	1	...	0	0	0	1	0	1	1	0	0	0
4	57	0	120	354	0	163	1	0.6	0	1	...	0	0	0	1	0	1	1	0	0	0
5 rows × 21 columns

result.drop('thalassemia_type_nothing',axis=1,inplace=True)
resultc = result.copy()
result.columns
Index(['age', 'sex', 'resting_blood_pressure', 'cholesterol',
       'fasting_blood_sugar', 'max_heart_rate_achieved',
       'exercise_induced_angina', 'st_depression', 'num_major_vessels',
       'target', 'chest_pain_type_atypical angina',
       'chest_pain_type_non-anginal pain', 'chest_pain_type_typical angina',
       'rest_ecg_type_left ventricular hypertrophy', 'rest_ecg_type_normal',
       'st_slope_type_flat', 'st_slope_type_upsloping',
       'thalassemia_type_normal', 'thalassemia_type_reversable defect',
       'thalassemia_type_fixed defect'],
      dtype='object')
X = result.drop('target', axis = 1)
          
y = result['target']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train=(X_train-np.min(X_train))/(np.max(X_train)-np.min(X_train)).values
X_test=(X_test-np.min(X_test))/(np.max(X_test)-np.min(X_test)).values
from sklearn.linear_model import LogisticRegression
logre = LogisticRegression()
logre.fit(X_train,y_train)
LogisticRegression()
y_pred = logre.predict(X_test)
actual = []
predcition = []

for i,j in zip(y_test,y_pred):
  actual.append(i)
  predcition.append(j) 

dic = {'Actual':actual,
       'Prediction':predcition
       }
result  = pd.DataFrame(dic)
import plotly.graph_objects as go
 
fig = go.Figure()
 
 
fig.add_trace(go.Scatter(x=np.arange(0,len(y_test)), y=y_test,
                    mode='markers+lines',
                    name='Test'))
fig.add_trace(go.Scatter(x=np.arange(0,len(y_test)), y=y_pred,
                    mode='markers',
                    name='Pred'))
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))
0.8688524590163934
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
              precision    recall  f1-score   support

           0       0.83      0.89      0.86        27
           1       0.91      0.85      0.88        34

    accuracy                           0.87        61
   macro avg       0.87      0.87      0.87        61
weighted avg       0.87      0.87      0.87        61

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_pred))
sns.heatmap(confusion_matrix(y_test,y_pred),annot=True)
[[24  3]
 [ 5 29]]
<AxesSubplot:>

from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.plot(fpr,tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for Heart disease classifier')
plt.xlabel('False positive rate (1-Specificity)')
plt.ylabel('True positive rate (Sensitivity)')
plt.grid(True)

import sklearn
sklearn.metrics.roc_auc_score(y_test,y_pred)
0.8709150326797386
print(logre.intercept_)
plt.figure(figsize=(10,12))
coeffecients = pd.DataFrame(logre.coef_.ravel(),X.columns)
coeffecients.columns = ['Coeffecient']
coeffecients.sort_values(by=['Coeffecient'],inplace=True,ascending=False)
coeffecients
[0.96772424]
Coeffecient
chest_pain_type_non-anginal pain	1.281633
chest_pain_type_typical angina	1.036787
max_heart_rate_achieved	1.028382
thalassemia_type_normal	0.754872
chest_pain_type_atypical angina	0.732086
rest_ecg_type_normal	0.338507
st_slope_type_upsloping	0.212089
rest_ecg_type_left ventricular hypertrophy	0.066738
thalassemia_type_fixed defect	0.009411
fasting_blood_sugar	-0.135551
age	-0.415560
cholesterol	-0.457060
st_slope_type_flat	-0.466004
resting_blood_pressure	-0.537316
thalassemia_type_reversable defect	-0.612915
exercise_induced_angina	-0.786479
sex	-1.124012
st_depression	-1.146936
num_major_vessels	-2.272170
<Figure size 720x864 with 0 Axes>
df.columns
Index(['age', 'sex', 'chest_pain_type', 'resting_blood_pressure',
       'cholesterol', 'fasting_blood_sugar', 'rest_ecg_type',
       'max_heart_rate_achieved', 'exercise_induced_angina', 'st_depression',
       'st_slope_type', 'num_major_vessels', 'thalassemia_type', 'target'],
      dtype='object')
df4 = df[df['target'] == 0 ][['age', 'sex', 'chest_pain_type', 'resting_blood_pressure',
       'cholesterol', 'fasting_blood_sugar', 'rest_ecg_type',
       'max_heart_rate_achieved', 'exercise_induced_angina', 'st_depression',
       'st_slope_type', 'num_major_vessels', 'thalassemia_type', 'target']] #target 0 - people with heart disease
plt.figure(figsize=(16,6))
sns.distplot(df4['max_heart_rate_achieved'])
C:\ProgramData\Anaconda2021\lib\site-packages\seaborn\distributions.py:2551: FutureWarning:

`distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).

<AxesSubplot:xlabel='max_heart_rate_achieved', ylabel='Density'>

plt.figure(figsize=(20,6))
sns.boxenplot(data=df4,x='rest_ecg_type',y='cholesterol',hue='st_slope_type')
<AxesSubplot:xlabel='rest_ecg_type', ylabel='cholesterol'>

plt.figure(figsize=(20,6))
sns.boxenplot(data=df4,x='chest_pain_type',y='max_heart_rate_achieved',hue='thalassemia_type')
<AxesSubplot:xlabel='chest_pain_type', ylabel='max_heart_rate_achieved'>

import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test,check_additivity=False)

shap.summary_plot(shap_values[1], X_test, plot_type="bar")

shap.summary_plot(shap_values[1], X_test)

def patient_analysis(model, patient):
  explainer = shap.TreeExplainer(model)
  shap_values = explainer.shap_values(patient)
  shap.initjs()
  return shap.force_plot(explainer.expected_value[1], shap_values[1], patient)
patients = X_test.iloc[3,:].astype(float)
patients_target = y_test.iloc[3:4]
print('Target : ',int(patients_target))
patient_analysis(model, patients)
Target :  0

patients = X_test.iloc[33,:].astype(float)
patients_target = y_test.iloc[33:34]
print('Target : ',int(patients_target))
patient_analysis(model, patients)
Target :  1

 y_test.iloc[10:11]
111    1
Name: target, dtype: int64
shap.dependence_plot('num_major_vessels', shap_values[1], X_test, interaction_index = "st_depression")

shap_values = explainer.shap_values(X_train.iloc[:50],check_additivity=False)
shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1], X_test.iloc[:50])


sample order by similarity

f(x)
plt.figure(figsize=(10,12))
coeffecients = pd.DataFrame(logre.coef_.ravel(),X.columns)
coeffecients.columns = ['Coeffecient']
coeffecients.sort_values(by=['Coeffecient'],inplace=True,ascending=False)
sns.heatmap(coeffecients,annot=True,fmt='.4f',cmap='Set1',linewidths=0.5)
<AxesSubplot:>


# This is my code about heart disease prediction 
