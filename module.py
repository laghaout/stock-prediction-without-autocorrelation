# -*- coding: utf-8 -*-
"""
Created on Fri Jun  24 20:43:59 2016

@author: Amine Laghaout
"""

def errorMetric(estimated, true):
    
    '''
    This function computes the average absolute difference between the 
    vectors of `estimated' and `true' values.
    '''
    
    from numpy import mean, array    
    
    return mean(abs(array(estimated) - array(true)))  

def classifier(algoName, algoParams):
    
    '''
    This function returns the algorithm `algoName' using the parameters
    `algoParams'.
    '''    
    
    if algoName == 'SGDRegressor':
        from sklearn.linear_model import SGDRegressor

    elif algoName == 'DecisionTreeRegressor':
        from sklearn.tree import DecisionTreeRegressor

    elif algoName == 'SVR':
        from sklearn.svm import SVR
        
    elif algoName == 'NuSVR':
        from sklearn.svm import NuSVR   

    elif algoName == 'Lasso':
        from sklearn.linear_model import Lasso

    elif algoName == 'RidgeCV':
        from sklearn.linear_model import RidgeCV
        
    elif algoName == 'Ridge':
        from sklearn.linear_model import Ridge

    elif algoName == 'ElasticNet':
        from sklearn.linear_model import ElasticNet

    elif algoName == 'LinearRegression':
        from sklearn.linear_model import LinearRegression

    elif algoName == 'LinearSVC':
        from sklearn.svm import LinearSVC

    elif algoName == 'RandomForestRegressor':
        from sklearn.ensemble import RandomForestRegressor

    elif algoName == 'GradientBoostingRegressor':
        from sklearn.ensemble import GradientBoostingRegressor

    else:
        print('ERROR: Invalid algorithm name', algoName)
    
    algo = eval(algoName+'('+algoParams+')')
        
    return algo

def evalEstimator(algo, df, predictors, targetName, CV_folds):

    '''
    This function returns the score and accuracy of a given algorithm `algo'
    applied to the predictors `predictors' a data frame `df'. The column of the
    data frame that acts as a target is labeled by `targetName'. Cross-
    validation is performed using `CV_folds' folds.
    '''

    from sklearn.cross_validation import KFold, cross_val_score
    from numpy import concatenate

    # Evaluate the score by cross-validation.
    scores = cross_val_score(algo, df[predictors], df[targetName], cv = CV_folds)

    kf = KFold(df.shape[0], n_folds = CV_folds, random_state = 1)
    predictions = []
    
    # For any given fold...
    for train, test in kf:

        # Select the predictors and corresponding targets for the current
        # fold.        
        train_predictors = (df[predictors].iloc[train,:])   
        train_target = df[targetName].iloc[train]
        
        algo.fit(train_predictors, train_target)
        
        # Make the predictors on the current test fold.
        test_predictions = algo.predict(df[predictors].iloc[test,:])
        predictions.append(test_predictions)
    
    # Concatenate the predictions made on all the folds.
    predictions = concatenate(predictions, axis = 0)       
    
    return (scores.mean(), errorMetric(list(predictions), list(df[targetName])))

def dataSynopsis(df, showSynopsis = True):
    
    '''
    This function prints to the screen a brief synopsis of the data in the 
    data frame `df' provided that showSynopsis is True.
    '''
    
    if showSynopsis:    
        print(df.head(5))
        print(df.describe())
        
        

def Plot2D(df, entryKey, predictors, targetName, xLabel = '', yLabel = '', 
           plotTitle = '', fontSize = 16, fontFamily = 'Times New Roman', 
           saveAs = None):

    ''' 
    Make a  two-dimensional plot of all the `predictors' as well as of of the
    target feature `targetName' as a function of the dates `entryKey' of the
    data frame `df'. The generated plot is saved as the file `saveAs'.
    '''

    import matplotlib.pyplot as plt
    import datetime as dt

    plt.rc('font', family = fontFamily)

    # Abscissa: Trading dates    
    x = [dt.datetime.strptime(d[:-5],'%m/%d/%Y').date() for d in df[entryKey]]    
    
    # Ordinates: Stock returns (either daily or compound)    
    yPredictors = df[predictors]

    # Plot the target feature
    yTarget = list(df[targetName])
    plt.plot_date(x, yTarget, 'r', linewidth=3, label = targetName) 
    
    # Plot the predictors
    for elem in yPredictors:
        if int(elem[1:]) < 9:
            lineStyle = '--'
        else:
            lineStyle = '-.'
        plt.plot_date(x, list(yPredictors[elem]), lineStyle, linewidth = 1, label=elem)

    plt.setp(plt.xticks()[1], rotation=30)   # Incline the date labels
    
    plt.xlabel(r'%s' % xLabel, fontsize = fontSize)
    plt.ylabel(r'%s' % yLabel, fontsize = fontSize)
    plt.legend(loc='upper left', bbox_to_anchor=(1,1))
    plt.title(r'%s' % plotTitle, fontsize = fontSize)
    plt.grid()
    
    if saveAs != None:
        plt.savefig(saveAs, bbox_inches='tight')
        
    plt.show()        
    plt.clf()  
    
    return df

def featureSelection(df, targetName, predictors = None, kBest = None, 
                     graphFeatures = False, fontSize = 16, plotTitle = '',
                     fontFamily = 'Times New Roman', saveAs = None):
    
    '''
    This function plots the relative relevance of the different predictors 
    `predictors' in a data frame `df' so as to best predict the target 
    `targetName'.
    
    Unless otherwise specified, all predictors in the data frame are 
    considered. Similarly, the selection of the best predictors can be limited 
    to the `kBest', otherwise all predictors are considered.
    
    The function returns the selection of best predictors.
    '''    
    
    from sklearn.feature_selection import SelectKBest, f_regression
    from numpy import log10
    
    # Extract the predictor labels if all are to be selected.
    if predictors == None:
        predictors = list(list(df.columns))
        if targetName in predictors:
            predictors.remove(targetName)
    
    # If all predictors are to be selected, select them all.
    if kBest == None:
        kBest = len(predictors)   
    
    selector = SelectKBest(f_regression, k = kBest)   
    selector.fit(df[predictors], df[targetName])   
    scores = -log10(selector.pvalues_)

    if graphFeatures is True:
        
        # Plot the p-values of the feature scores.
        import matplotlib.pyplot as plt
        
        plt.rc('font', family = fontFamily)
        plt.title(plotTitle, fontsize = fontSize)
        plt.bar(range(len(predictors)), scores, align = 'center')
        plt.ylabel('p-values', fontsize = fontSize)
        plt.xticks(range(len(predictors)), predictors, rotation = 'vertical', 
                   fontsize = fontSize)
        plt.xlim([-.5,len(predictors)-.5])   
        
        if saveAs != None:
            plt.savefig(saveAs, bbox_inches = 'tight')
        
        plt.show()
        plt.clf()   
    
    return [elem for idx, elem in enumerate(predictors) if selector.get_support()[idx]]    

def paramSearch(features, targets, algoName, param_grid,  
                CV_folds = 3):

    '''    
    This function performs a parameter sweep over the array of parameters 
    `param_grid' of the algorithm `algoName'. It returns the highest-scoring 
    combination of these parameters based on the `score'. Cross-validation is 
    performed on the `features' and `targets' using `CV_folds' folds.
    '''

    from sklearn.grid_search import GridSearchCV
    
    estimator = classifier(algoName, '')

    algo = GridSearchCV(estimator, param_grid, 
                        cv = CV_folds, scoring = None)
    algo.fit(features, targets)     
     
    return algo.best_params_

def exportPrediction(algo, df, df_test, predictors, targetName, entryKey, 
                     predictFile):
    
    '''
    This function exports the predictions on the test set `df_test' based on 
    the training of estimator `algo' on the `predictors' in the feature set 
    `df'. For each entry, the key is labeled by `entryKey' and the target by
    `targetName'. The predictions (i.e., the combination of the key and the 
    predicted target value) are saved in the file `predictFile.
    '''    
    
    from pandas import DataFrame                
    
    algo.fit(df[predictors], df[targetName])
    predictions = algo.predict(df_test[predictors])
    
    submission = DataFrame({'Date': df_test[entryKey], 
                            'Value': predictions})
    
    submission = submission[['Date', 'Value']] 
    
    submission.to_csv(predictFile, index = False)

    return submission                     

def fillMissing(dfs, feature, val):
    
    '''
    This function fills the missing features labeled by `feature' in all the 
    data frames contained in the tuple of data frames `dfs'. The filling values 
    is specified by `val' and can also be either the median or the mean of the 
    already-present values by setting `val' to `median' and `mean',
    respectively.
    '''

    import pandas

    for df in dfs:
        
        # Check whether the tuple consists of a single data frame. If so, do 
        # not loop through the elements of the data frame.
        if type(dfs) is pandas.core.frame.DataFrame:
            df = dfs
        
        if val == 'median':
            df[feature] = df[feature].fillna(df[feature].median())
        elif val == 'mean':
            df[feature] = df[feature].fillna(df[feature].mean())
        else:
            df[feature] = df[feature].fillna(val)
            
        if type(dfs) is pandas.core.frame.DataFrame:
            break           
            
def remapFeature(dfs, feature, mapping = None):
    
    '''
    This function maps the feature `feature' in the tuples of data frames `dfs'  
    according to the `mapping' which consists of two lists whereby the first is 
    to have its elements mapped to the second. If `mapping' is None, the set of 
    unique elements is extracted and each element is assigned an integer 
    starting from zero.
    
    WARNING: `dfs' should contain at least two data frames.
    '''

    import pandas as pd
    
    # If the mapping is not specified, extract all the unique values of the 
    # features from the union of all data frames and map them into the natural
    # numbers starting at zero.
    if mapping == None:
        
        # Check whether we have a single data frame or a list.
        if type(dfs) is pd.core.frame.DataFrame:
            df = dfs
        # If it is a list, concatenate all the data frames into one.
        else:
            df = pd.concat(dfs)
            
        fromList = df[feature].unique()
        toList = list(range(len(fromList)))
        
    else:
        
        fromList, toList = mapping

    # TO-DO: Double-check why this is necessary
    # http://stackoverflow.com/questions/37890845/is-the-object-data-type-static-in-python
    toList = pd.Series(range(len(fromList)), index = fromList)

    for df in dfs:
        
        df[feature] = df[feature].map(toList)
    
def numericFeatures(dfs):
    
    '''
    This function returns the labels of the non-numeric features in the list of
    data frames `dfs'. 
    '''
    
    import pandas as pd
    
    # Check whether we have a single data frame or a list.
    if type(dfs) is pd.core.frame.DataFrame:
        df = dfs
        
    # If it is a list, concatenate all the data frames into one.
    else:
        df = pd.concat(dfs)
        
    df = df.select_dtypes(include = ['number'])
        
    return list(df.columns)
    
def removeLowVar(features, varThreshold = 0):
    
    '''
    This function returns a list of predictors from `features' that have a 
    variance above `varThreshold'.
    
    TO-DO: `features' should be a tuple so that it can emcompass both the 
    training and test features.
    '''
    
    from sklearn.feature_selection import VarianceThreshold
    selector = VarianceThreshold(varThreshold)
    selector.fit_transform(features)
        
    predictors =  [elem for idx, elem in enumerate(features.columns) if selector.get_support()[idx]]      
            
    return predictors