import pandas as pd

from src.preprocessing import FeaturesPreprocessing
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.decomposition import PCA

    
def train_evaluate(X, y, features_selection=[], pipeline_positions=[]):
    """
    Compute the training and test F1 scores using cross validation for several classification models, with the option of adding feature selection
    processes to the pipeline.

    Inputs:
        X[pandas.DataFrame] : dataframe containing cleaned features
        y[pandas.DataFrame] : containing containing cleaned target class
        Optionnal :
            features_selection[List[sklearn]] : list of feature selection methods
            pipeline_positions[List[Int64]] : list of indices referring to the position of feature selection method in the pipeline 
    Output:
        scores_df[pandas.DataFrame] : dataframe containing the training and test F1 scores of all models
    """

    # Initialize all the models
    SVM_RBF = SVC()
    SVM_Linear = SVC(kernel='linear')
    SVM_Poly2 = SVC(kernel='poly',degree=2)
    SVM_Poly3 = SVC(kernel='poly',degree=3)
    KNN3 = KNeighborsClassifier(n_neighbors=3,weights='distance')
    KNN3_u = KNeighborsClassifier(n_neighbors=3, weights='uniform')
    KNN8 = KNeighborsClassifier(n_neighbors=8,weights='distance')
    KNN8_u = KNeighborsClassifier(n_neighbors=8,weights='uniform')
    Naive_Bayes = GaussianNB()
    LogReg = LogisticRegression()
    Tree = DecisionTreeClassifier()
    Forest = RandomForestClassifier()

    # List all the models and their name
    models = [SVM_RBF, SVM_Linear, SVM_Poly2,SVM_Poly3,KNN3,KNN3_u,KNN8,KNN8_u,Naive_Bayes,LogReg,Tree,Forest]
    names = ["SVM RBF", "SVM Linear", "SVM Poly2", "SVM Poly3", "KNN3", "KNN3 uni", "KNN8", "KNN8 uni", "Naive Bayes", "Logistic Regression", "Decision Tree", "Random Forest"]

    # Initialize the score dataframe
    scores_df = pd.DataFrame(columns=['model name', 'training score', 'test score'])

    # Loop over each model
    for (model, model_name) in zip(models,names):

        # Define the pipeline integrating the preprocessing, training and evaluation stages
        pipeline = Pipeline(steps=[
            ('preprocessing', FeaturesPreprocessing()),
            ('model', model)
            ])
        
        # Add the feature selection processes in the pipeline, if they exist
        if len(features_selection) > 0 and len(pipeline_positions) > 0:

            for (feature_selection, index) in zip(features_selection, pipeline_positions):

                pipeline.steps.insert(index, feature_selection)
        
        # Calculate training and test F1 scores using cross validation
        results = cross_validate(pipeline, X, y, cv=3, scoring='f1', return_train_score=True)
        scores_df.loc[scores_df.shape[0]] = [model_name, results['train_score'].mean(), results['test_score'].mean()]
    
    return scores_df



def pca_train_test(X, y, model_name):
    """
    Compute and plot the training and test metric (accuracy, recall, f1_score) with respect to the number of principal component.

    Inputs:
        X[pandas.DataFrame] : dataframe containing cleaned features
        y[pandas.DataFrame] : containing containing cleaned target class
        model_name[String] : name of the classifier model
    Output:
        pca_scores_df[pandas.DataFrame] : DataFrame containing the model’s test F1 scores as a function of the number of principal components. 
    """

    # Define the classification model
    if model_name == 'SVM RBF':
        model = SVC()
    elif model_name == 'KNN3':
        model = KNeighborsClassifier(n_neighbors=3,weights='distance')
    elif model_name == 'Naive Bayes':
        model = GaussianNB()
    elif model_name == 'Logistic Regression':
        model = LogisticRegression()
    elif model_name == 'Decision Tree':
        model = DecisionTreeClassifier()
    elif model_name == 'Random Forest':
        model = RandomForestClassifier()

    # Store the training and test scores
    pca_scores_df = []

    # Loop over the number of PCA components
    for i in range(X.shape[1]+1):

        if i == 0 :
            pipeline = Pipeline(steps=[
                ('preprocessing', FeaturesPreprocessing()),
                ('model', model)
            ])
        
        else :
            pipeline = Pipeline(steps=[
                ('preprocessing', FeaturesPreprocessing()),
                (f'PCA | {i} components', PCA(n_components=i)),
                ('model', model)
            ])
        
        # Calculate the test f1 score using i principal components
        test_score_i = cross_validate(pipeline, X, y, cv=5, scoring='f1')['test_score'].mean()

        pca_scores_df.append(test_score_i)

    return pca_scores_df



