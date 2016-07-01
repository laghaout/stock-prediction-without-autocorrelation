# -*- coding: utf-8 -*-
"""
Created on Fri Jun  24 10:57:14 2016

@author: Amine Laghaout
"""

import pandas as pd
import module as md

#%% SETTINGS

#%% Files

trainFile = 'data/input.csv'    		# Data file
predictFile = 'data/predictions.csv'            # Prediction file

#%% Key and target labels

keyName = 'date'    # Key of the data entries
targetName = 'S1'   # Label of the target

#%% Algorithm

# This dictionary of algorithms allows them to be retrieved using 
# abbreviations.
algos = {\
'linSVR': ('SVR', 'kernel="linear", C = .02, epsilon = .22', None), \
'SGDR': ('SGDRegressor', None, None), \
'Ridge': ('Ridge', 'alpha = 11', None)}

algoName = 'linSVR' # Abbreviation for the algorithm used
CV_folds = 3        # Number of folds for the cross-validation
kBest = 6           # Number of best features to select

#%% Misc

showSynopsis = False    # Show the data synopsis?
showFeatures = True     # Show feature relevance?
showEvolution = True    # Show the evolution plots for the returns?
decDigits = 4           # Number of decimal digits to display
num_train_data = 50     # Number of training data
varThreshold = 0        # Minimum variance required of the features
halfLife = 1            # Half-life (in days) of the exponential moving average

#%% IMPORT DATA

trainFeatures = pd.read_csv(trainFile)  
trainFeatures = trainFeatures.dropna(subset = [keyName]) # Remove void fields
predictors = list(trainFeatures.columns)        # Extract feature names
testFeatures = trainFeatures[num_train_data:]   # Training data
trainFeatures = trainFeatures[:num_train_data]  # Testing data

#%% FEATURE ENGINEERING

#%% Initial synopsis

md.dataSynopsis(trainFeatures, showSynopsis)

#%% Remove the key and the target from the list of predictors.

predictors = [k for k in predictors if k not in [keyName, targetName]]
#print('Predictors after removal of key and target:\n', predictors)

#%% Remove low-variance features

predictors = md.removeLowVar(trainFeatures[predictors], varThreshold)
#print('Predictors after removal of low-variance features:\n', predictors)


#%% Add new features: cumulative return and moving average return

from numpy import cumprod

ewma = pd.ewma      # Exponentially-weighted moving averages

CR_predictors = []  # Cumulative returns
MA_predictors = []  # Moving average returns

# For eacher Japanese stock---i.e., for each predictor---create a two new ones: 
# One based on its cumulative return, assuming the stock starts at an arbitrary
# 100 unit value, and another one based on its exponentially-weighted moving
# average with a half-life of `halfLife'.

for p in predictors:
    
    # Compute the compound return (CR)
    CR_feature = 'CR '+p
    trainFeatures[CR_feature] = cumprod((1+trainFeatures[p][:]/100))  
    testFeatures[CR_feature] = cumprod((1+testFeatures[p][:]/100))  
    CR_predictors.append(CR_feature)   
    
    # Compute the moving average (MA) returns
    MA_feature = 'MA '+p
    EMOV_n = ewma(trainFeatures[p], halflife = halfLife)
    trainFeatures[MA_feature] = list(EMOV_n)
    EMOV_n = ewma(testFeatures[p], halflife = halfLife)
    testFeatures[MA_feature] = list(EMOV_n)        
    MA_predictors.append(MA_feature)

# Select the `kBest' predictors among the cumulative return and moving average
# features. Plot their relevance using the p-value.

CR_predictors = md.featureSelection(trainFeatures, targetName, CR_predictors, 
                                    kBest, showFeatures, 
                                    saveAs = 'CompoundReturnRelevance.pdf',
                                    plotTitle = 'Compound return')

MA_predictors = md.featureSelection(trainFeatures, targetName, MA_predictors, 
                                    kBest, showFeatures, 
                                    saveAs = 'MovingAverageRelevance.pdf',
                                    plotTitle = 'Exponential moving average return')

# %% Plot the evolution of the stocks

if showEvolution:
    
    from numpy import cumprod
    
    CR = trainFeatures.copy()
    
    # Compute the cumulative return
    for p in [targetName]+predictors:
        CR[p] = 100*cumprod((1+trainFeatures[p][:]/100))  

    CR = md.Plot2D(CR, keyName, predictors, targetName, 
                   saveAs = 'CompoundReturn.pdf', plotTitle = 'Compound return',
                   yLabel = 'Percentage')
    DR = md.Plot2D(trainFeatures, keyName, predictors, targetName, 
                   saveAs = 'DailyReturn.pdf', plotTitle = 'Daily return',
                   yLabel = 'Percentage')    

#%% Select the best predictors

predictors = md.featureSelection(trainFeatures, targetName, predictors, kBest,
                                 showFeatures, plotTitle = 'Daily return',
                                 saveAs = 'DailyRelevance.pdf')
#print('Predictors after ', kBest,'-best feature selection:\n', predictors, sep = '')

#%% Final synopsis

md.dataSynopsis(trainFeatures[predictors], showSynopsis)
print('The predictors are:', predictors)

#%% CLASSIFICATION

#%% Preliminary classification

# Produce the estmation algorithm
(algoName, algoParams, algoPredictors) = algos[algoName]
if algoPredictors != None:
    predictors = algoPredictors
if algoParams is None:
    algoParams = ''
algo = md.classifier(algoName, algoParams)

# Evaluate the performance of the algorithm
(scores, error) = md.evalEstimator(algo, trainFeatures, predictors, targetName, CV_folds)
print(algoName, ': ', sep = '')
print('\tScore\t', round(scores, decDigits), '\n\tError\t', round(error, decDigits))




#%% Optimization by grid search
'''
param_grid = [{'C': [.01,.02,.03], 'epsilon': [.19,.20,.21,.22,.23,.24]}]

bestParams = md.paramSearch(trainFeatures[predictors], 
                            trainFeatures[targetName], algoName, param_grid, \
                            CV_folds = CV_folds)
print('The best parameters are', bestParams)
'''

#%% PREDICTION

submission = md.exportPrediction(algo, trainFeatures, testFeatures, predictors, 
                                 targetName, keyName, predictFile)

testFeatures[targetName] = submission['Value']
predictors = list(trainFeatures.columns)[2:11]

if showEvolution:
    
    from numpy import cumprod
    
    offsetCR = CR.copy()        # Offset: last compound value in training set
    CR = testFeatures.copy()
    
    # Compute the cumulative return starting where we left off at the end of
    # the training data.
    for p in [targetName]+predictors:
        CR[p] = float(offsetCR[p][-1:])*cumprod((1+testFeatures[p][:]/100))
        
    CR = md.Plot2D(CR, keyName, predictors, targetName, 
                   saveAs = 'ProjectedCompoundReturn.pdf', 
                   plotTitle = 'Projected compound return',
                   yLabel = 'Percentage')
    DR = md.Plot2D(testFeatures, keyName, predictors, targetName, 
                   saveAs = 'ProjectedDailyReturn.pdf', 
                   plotTitle = 'Projected daily return',
                   yLabel = 'Percentage')  
                   