import numpy as np
import pandas as pd

from ml_inner_cross_val import * #source my own file

#=====change these=====

annot_dict={
			'Pwy':["DENOVOPURINE2-PWY","BRANCHED-CHAIN-AA-SYN-PWY","TCA"],
			'pcomplex':["CPLX0-7452","NADH-DHI-CPLX","CPLX0-3382"],
			'operon':["TU00305","TU00201","TU0-13499"],
			'kegg_modules':["eco_M00009\xa0\xa0Citrate cycle (TCA cycle, Krebs cycle)","eco_M00048\xa0\xa0Inosine monophosphate biosynthesis, PRPP + glutamine => IMP","eco_M00570\xa0\xa0Isoleucine biosynthesis, threonine => 2-oxobutanoate => isoleucine"],
			'regulator_name':["LexA","PhoP","ModE"]
}

annot_map_dict={ 
			'Pwy':{'BRANCHED-CHAIN-AA-SYN-PWY':0,'DENOVOPURINE2-PWY':1,'TCA':2},
			'pcomplex':{'CPLX0-3382':0,'CPLX0-7452':1,'NADH-DHI-CPLX':2},
			'operon':{'TU0-13499':0,'TU00201':1,'TU00305':2},
			'kegg_modules':{'eco_M00009\xa0\xa0Citrate cycle (TCA cycle, Krebs cycle)':0,'eco_M00048\xa0\xa0Inosine monophosphate biosynthesis, PRPP + glutamine => IMP':1,'eco_M00570\xa0\xa0Isoleucine biosynthesis, threonine => 2-oxobutanoate => isoleucine':2},
			'regulator_name':{'LexA':0,'ModE':1,'PhoP':2}
}



cv=3
test_size=0.2
random_state=101


#======================



phenotype=pd.read_csv('Nich_Price_quantitative.csv',header=None,index_col=0)
phenotype=phenotype.reset_index() #now the ids are on the 1st column (type: int64)
name=['ids']
name.extend(list(range(1,486+1)))
name
phenotype.columns=name

import pyreadr
result = pyreadr.read_r('id_allAttributes.RData')
id_allAttributes=result['id_allAttributes']
id_allAttributes=id_allAttributes.astype({'ids': 'int64'})

df=phenotype.merge(id_allAttributes,on="ids",how='left')




result_list=[]
for annot in ['Pwy','pcomplex','operon','regulator_name','kegg_modules']:
	id_annot=df[['ids',annot]].drop_duplicates()
	id_annot.columns=['ids','annot']
	id_annot_table=pd.DataFrame(id_annot.iloc[:,1].value_counts())
	
	annot_list=annot_dict[annot]
	id_selected=id_annot.query('annot in @annot_list').sort_values(by=['annot'])

	id_selected_phenotype=id_selected.merge(phenotype,on="ids",how='left')
	X=id_selected_phenotype.iloc[:,2:]
	#normalize the features
	from sklearn import preprocessing
	min_max_scaler = preprocessing.MinMaxScaler()
	X = pd.DataFrame(min_max_scaler.fit_transform(X))
	X_Nich_only=X.iloc[:,0:324]

	y=np.array(id_selected.annot)
	#To ensure y labels are interpreted correctly, I converted them to 0~2
	#tf.keras.utils.to_categorical has this problem: if y=[6,9,10], there will be dummy variables 0~10, not just 0~2 (https://stackoverflow.com/questions/41494625/issues-using-keras-np-utils-to-categorical/43314437)
	map_=annot_map_dict[annot]
	y=np.array([map_[class_] for class_ in y])
	
	
	#logistic regression
	Nich_Price=my_logistic_regression(X,y,cv=cv)
	Nich=my_logistic_regression(X.iloc[:,0:324],y,cv=cv)
	result_list.append(Nich_Price)
	result_list.append(Nich)
	
	#decision tree
	param_grid={}
	Nich_Price=my_decision_tree(X,y,cv,param_grid)
	Nich=my_decision_tree(X.iloc[:,0:324],y,cv,param_grid)
	result_list.append(Nich_Price)
	result_list.append(Nich)
	
	
	#random forest
	max_depth = [1,5,10,100]
	criterion=['entropy','gini']
	n_estimators=[10,25,50,100,200] #this can possibly be set to a much higher value but it would take lots of time
	param_grid={'criterion':criterion,'max_depth':max_depth,'n_estimators':n_estimators}

	Nich_Price=my_random_forest(X,y,cv,param_grid,n_jobs=-1)
	Nich=my_random_forest(X.iloc[:,0:324],y,cv,param_grid,n_jobs=-1)
	result_list.append(Nich_Price)
	result_list.append(Nich)

	#boosting
	n_estimators = [5,10,15,20,50,100,200,400]
	max_iter=10000
	max_depth=[1,2,3,10,20,50]
	param_grid={'n_estimators':n_estimators,'max_depth':max_depth}

	Nich_Price=my_boosting(X,y,cv,param_grid)
	Nich=my_boosting(X.iloc[:,0:324],y,cv,param_grid)
	result_list.append(Nich_Price)
	result_list.append(Nich)
	
	#svm
	ds = [1,2,3,4]
	Cs = [0.001, 0.01, 0.1, 1, 10,100,1000] #ref: https://medium.com/@aneesha/svm-parameter-tuning-in-scikit-learn-using-gridsearchcv-2413c02125a0
	kernels=['linear', 'poly', 'rbf', 'sigmoid'] # I removed 'precomputed' kernel because it only accepts data that looke like: (n_samples, n_samples). Ref: https://stackoverflow.com/questions/36306555/scikit-learn-grid-search-with-svm-regression/36309526
	param_grid={'degree':ds,'C':Cs, 'kernel':kernels}

	Nich_Price=my_svm(X,y,cv,param_grid)
	Nich=my_svm(X.iloc[:,0:324],y,cv,param_grid)
	result_list.append(Nich_Price)
	result_list.append(Nich)
	
	#neuro net
	from keras.layers import Dense 
	from keras.models import Sequential
	from keras.utils import to_categorical
	from sklearn.model_selection import train_test_split
	from sklearn.model_selection import cross_val_score


	def my_neuronet(X,y):
		test_size = 0.2
		random_state=123
		epochs=1000
		optimizer='sgd'
		loss='categorical_crossentropy'



		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state=random_state,stratify=y) #not sure if stratify=y works when there are more than 2 classes for the labels

		model = Sequential()
		model.add(Dense(300,activation='relu',input_dim=X.shape[1])) #X.shape[1] should equal the number of features
		model.add(Dense(300,activation='relu'))
		model.add(Dense(300,activation='relu'))
		model.add(Dense(len(np.unique(y_train)),activation='softmax')) #the number of the output layer should equal the number of unique outcomes (response variables)

		model.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])

		model.fit(X_train, to_categorical(y_train),epochs=epochs,verbose=0) 

		y_pred = model.predict(X_test)
		y_pred = np.argmax(y_pred,axis=1)

		#print("accuracy = ",np.mean(y_test==y_pred))
		#print("precision= ",precision_score(y_test,y_pred,average='weighted') )
		return([{},np.mean(y_test==y_pred),precision_score(y_test,y_pred,average='weighted')])
    
    
	Nich_Price=my_neuronet(X,y)
	Nich=my_neuronet(X.iloc[:,0:324],y)
	result_list.append(Nich_Price)
	result_list.append(Nich)


	

best_param=[metrics[0] for metrics in result_list]
accuracy=[metrics[1] for metrics in result_list]
precision=[metrics[2] for metrics in result_list]	


best_param_df = pd.DataFrame(np.array(best_param).reshape(12,5,order='F'))
best_param_df.to_csv('best_param_df.csv')

accuracy_df = pd.DataFrame(np.array(accuracy).reshape(12,5,order='F'))
accuracy_df.to_csv('accuracy_df.csv')

precision_df = pd.DataFrame(np.array(precision).reshape(12,5,order='F'))
precision_df.to_csv('precision_df.csv')








	
	