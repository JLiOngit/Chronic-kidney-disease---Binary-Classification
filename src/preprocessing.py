import pandas as pd
import numpy as np

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer


def cleaning(df):
    """
    Clean the initial dataframe with the following steps:
        1) Clean the categorical columns
        2) Convert the misclassified categorical features into numerical features

    Input:
        df[pd.DataFrame] : the raw input data
    Output:
        cleaned_df[pd.DataFrame] : the cleaned dataframe
    """

    # Copy the initial dataframe
    cleaned_df = df.copy()

    # Drop the non-informative column 'id'
    cleaned_df = cleaned_df.drop('id', axis=1)

    # Clean the categorical columns
    categorical_columns = cleaned_df.select_dtypes('object').columns
    for column in categorical_columns:
        cleaned_df[column] = cleaned_df[column].str.strip()
        cleaned_df[column] = cleaned_df[column].replace('?', np.nan)
    
    # Retrieve the misclassified categorical features
    misclassified_features = [column for column in categorical_columns if cleaned_df[column].nunique() > 10]

    # Retrieve the actual categorical features
    categorical_features = categorical_columns.difference(misclassified_features)
    try:
        assert len(misclassified_features) + len(categorical_features) == len(categorical_columns)
    except AssertionError:
        print("Error in features that are considered as misclassified.")

    # Convert the misclassified categorical features into numerical features
    for feature in misclassified_features:
        try:
            cleaned_df[feature] = pd.to_numeric(cleaned_df[feature], errors='raise')
        except ValueError:
            print(f"The misclassified feature {feature} cannot be converted into numerical type.")

    # Retrieve the actual numerical features
    numerical_features = cleaned_df.select_dtypes(['Int64', 'Float64']).columns
    try:
        assert len(numerical_features) + len(categorical_features) == cleaned_df.shape[1]
    except AssertionError:
        print(f'Numerical ({len(numerical_features)}) and categorical ({len(categorical_features)}) features do not match the total number of features ({cleaned_X.shape[1]})')

    # Label-encode the target class
    cleaned_df['classification'] = cleaned_df['classification'].map({'ckd': 1, 'notckd': 0})

    return cleaned_df



class FeaturesPreprocessing(TransformerMixin, BaseEstimator):
    """
    Preprocess features of the cleaned dataframe with the following steps:
        1) For categorical features :
            - handle missing values by imputing the most frequent value.
            - Compute one-hot-encoding
        2) For numerical value :
            - Compute standardization to avoid favorizing numerical feature with a wider spread in KNN method
            - handle missing values using KNN.
    """


    def __init__(self, scaler=True):
        self.scaler = scaler
        self.categorical_features_ = None
        self.numerical_features_ = None
        self.preprocessing_ = None
        self.column_names_ = None


    def fit(self, X, y=None):

        cleaned_X = X.copy()

        self.categorical_features_ = cleaned_X.select_dtypes('object').columns
        self.numerical_features_ = cleaned_X.select_dtypes(['Int64', 'Float64']).columns

        # For categorical features : impute the most reccurent value and compute one-hot-encoding
        categorical_preprocessing = Pipeline(steps=[
            ('simple imputer', SimpleImputer(strategy='most_frequent')),
            ('one-hot-encoder', OneHotEncoder(drop='if_binary', sparse_output=False)),
        ])

        # # For numerical features : impute a value using KNN method and optionally scale data
        if self.scaler:
            numerical_preprocessing = Pipeline(steps=[
                ('scaler', StandardScaler()),
                ('KNN imputer', KNNImputer(weights='distance', n_neighbors=8)),
            ])
        else:
            numerical_preprocessing = Pipeline(steps=[
                ('KNN imputer', KNNImputer(weights='distance', n_neighbors=8)),
            ])
        

        # Apply the preprocessing process
        self.preprocessing_ = ColumnTransformer(transformers=[
            ('categorical preprocessing', categorical_preprocessing, self.categorical_features_),
            ('numerical preprocessing', numerical_preprocessing, self.numerical_features_)
        ])
    
        self.preprocessing_.fit(cleaned_X)

        cat_cols_transformed = self.preprocessing_.named_transformers_['categorical preprocessing']['one-hot-encoder'].get_feature_names_out(self.categorical_features_)
        self.column_names_ = list(self.categorical_features_) + list(self.numerical_features_)

        return self
    
    
    def transform(self, X):
        cleaned_X = X.copy()
        X_transformed = self.preprocessing_.transform(cleaned_X)
        return pd.DataFrame(X_transformed, columns=self.column_names_, index=X.index)






