#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot as plt
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi',
                 'salary',
                 'deferral_payments', 
                 'total_payments',
                 'loan_advances',
                 'bonus', 
                 'restricted_stock_deferred', 
                 'deferred_income',
                 'total_stock_value',
                 'expenses', 
                 'exercised_stock_options',
                 'other', 
                 'long_term_incentive',
                 'restricted_stock',
                 'director_fees',
                 'to_messages', 
                 'from_poi_to_this_person',
                 'from_messages',
                 'from_this_person_to_poi',
                 'shared_receipt_with_poi'
                 ] 

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)
	
print(data_dict.keys())	
### Quantidade de amostras
print("Total de amostras", len(data_dict))	

### Quantidade de POIs e não POIs
POI = 0
NPOI = 0

for i in  data_dict.keys():
    if(data_dict[i]["poi"]):
	    POI +=1
    else:
	    NPOI +=1	

print("Total de POIs", POI)
print("Total de não POIs", NPOI)
print("Total de features", len(data_dict['METTS MARK'].values()))

### Características com muitos valores faltantes
### Identificar amostras que possuem muitos atributos nulos

data_remove = []
for i in data_dict:
    if data_dict[i]['salary'] == 'NaN' and data_dict[i]['bonus'] == 'NaN' and data_dict[i]['director_fees'] == 'NaN' and data_dict[i]['total_stock_value'] == 'NaN' and data_dict[i]['expenses'] == 'NaN' :
	    data_remove.append(i)
		
print(data_remove)		

#Características das amostras
#print("WODRASKA JOHN", data_dict["WODRASKA JOHN"])	
#print("LOCKHART EUGENE E", data_dict["LOCKHART EUGENE E"])	
#print("THE TRAVEL AGENCY IN THE PARK'", data_dict["THE TRAVEL AGENCY IN THE PARK"])	
	

### Task 2: Remove outliers
#visualização gráfica
salary = []
bonus = []
for i in data_dict:
    salary.append(data_dict[i]["salary"])
    bonus.append(data_dict[i]["bonus"])
	
x = list(range(1, len(data_dict)+1))
plt.scatter( x, salary )	
plt.ylabel('Salário')
plt.show()

plt.scatter( x, bonus )	
plt.ylabel('Bônus')
plt.show()

#Identificar amostra que é um outlier
for i in data_dict:
    if data_dict[i]["salary"] != 'NaN' and data_dict[i]["salary"] > 2000000:
        name_key = i
	
##A amostra 'Total' representa a soma total de todas as colunas	
print(name_key)	
data_remove.append(name_key)

#remover os outliers do conjunto de dados
for i in data_remove:
    data_dict.pop(i, 0)
	
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
list_frac_email_from_poi = []

for i in data_dict:

    from_poi_to_this_person = data_dict[i]["from_poi_to_this_person"]
    to_messages = data_dict[i]["to_messages"]
    if from_poi_to_this_person == 'NaN' or to_messages == 'NaN':
	    list_frac_email_from_poi.append(0)
    elif to_messages > 0:
	    list_frac_email_from_poi.append(from_poi_to_this_person/to_messages)
	    
#print(list_frac_email_from_poi)

list_frac_email_to_poi = []

for i in data_dict:

    from_this_person_to_poi = data_dict[i]["from_this_person_to_poi"]
    from_messages = data_dict[i]["from_messages"]
    if from_this_person_to_poi == 'NaN' or from_messages == 'NaN':
	    list_frac_email_to_poi.append(0)
    elif from_messages > 0:
	    list_frac_email_to_poi.append(from_this_person_to_poi/from_messages)
	    
#print(list_frac_email_to_poi)

t_total_stock_value = 0
for i in data_dict:
    if data_dict[i]['total_stock_value'] != 'NaN':
        t_total_stock_value = t_total_stock_value + data_dict[i]['total_stock_value']

total_stock_value_percent = []
for i in data_dict:
    total_stock_value_n = data_dict[i]["total_stock_value"]
    if total_stock_value_n !=  'NaN':
	    total_stock_value_percent.append(total_stock_value_n / t_total_stock_value)
    else:
        total_stock_value_percent.append(0)	
	    
#print(total_stock_value_percent)

# add this new feature to my data

for n, i in enumerate(data_dict):
    data_dict[i]['fraction_from_poi'] = list_frac_email_from_poi[n]
    data_dict[i]['fraction_to_poi'] = list_frac_email_to_poi[n]
    data_dict[i]['fraction_total_stock_value'] = total_stock_value_percent[n]

features_list += ['fraction_from_poi', 'fraction_to_poi', 'fraction_total_stock_value']

print("Total de features", len(features_list))
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

###Seleção de features

best_features = SelectKBest(k=12)
best_features.fit(features, labels)

list_best_features = []
for n, i in enumerate(features_list[1:]):
    list_best_features.append({"feature": i, "score": best_features.scores_[n]}) 
	
newlist = sorted(list_best_features, key=lambda k: k['score'], reverse=True) 
#print(newlist)


features_list_new = []
for i in newlist[:12]:
    features_list_new.append(i["feature"])
features_list_new.insert(0, 'poi')	
print(features_list_new)	

###labels e features spmene com as 12 features selecionadas
data = featureFormat(data_dict, features_list_new)
labels, features = targetFeatureSplit(data)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

from sklearn.ensemble import RandomForestClassifier

parameters= {
            "criterion": ['entropy', 'gini'],
            "n_estimators": [25, 50, 75, 100, 200],
            "bootstrap": [False, True],
            "max_features": ['auto', 'sqrt', 'log2', 0.1, 0.2]
            }

clf_rfc = RandomForestClassifier(random_state=0)
clf_rfc_gs = GridSearchCV(clf_rfc, parameters, cv=5)

from sklearn.tree import DecisionTreeClassifier

parameters= {'min_samples_split' : range(10,500,20),
            "criterion": ["gini", "entropy"]}
clf_tree= DecisionTreeClassifier(random_state=0)
clf_tree= GridSearchCV(clf_tree,parameters, cv=5)


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.15, random_state=42)
	
clf_rfc_gs.fit(features_train, labels_train)	

print("RandomForestClassifier:")
print("Best Params", clf_rfc_gs.best_params_)

pred1 = clf_rfc_gs.predict(features_test)
acc = accuracy_score(labels_test, pred1)
print('Accuracy: ' + str(acc))
print('Precision: ', precision_score(labels_test, pred1))
print('Recall: ', recall_score(labels_test, pred1))

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
	
clf_tree.fit(features_train, labels_train)
pred2 = clf_tree.predict(features_test)	

print("DecisionTreeClassifier:")
print("Best Params", clf_tree.best_params_)

acc = accuracy_score(labels_test, pred2)
print('Accuracy: ' + str(acc))
print('Precision: ', precision_score(labels_test, pred2))
print('Recall: ', recall_score(labels_test, pred2))
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

###Método final utilizdo com os parâmetros definidos pelo gridSearch
clf= RandomForestClassifier(bootstrap= False, criterion= 'entropy', max_features= 0.2, n_estimators= 50)
dump_classifier_and_data(clf, my_dataset, features_list_new)