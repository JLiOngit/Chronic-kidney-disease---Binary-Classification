import matplotlib.pyplot as plt
import seaborn as sns

def numerical_distribution(df, numerical_features, n_rows, n_cols):
    """
    Plot the distributions of multiple numerical features in a grid of subplots, including Kernel Density Estimates (KDE) 
    and the proportion of missing values for each feature.

    Inputs:
        df[pandas.DataFrame] : the dataframe containing the data
        numerical_features[list] : list of numerical feature names
        n_rows[Int] : number of rows in the subplot grid.
        n_cols[Int] : number of columns in the subplot grid.
    Outputs:
        figure, axes
    """

    # Create a figure with a grid of subplots
    figure, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(40,50))

    axes = axes.flatten()

    # Define the title of the figure
    figure.suptitle('Distribution of numerical features', fontsize=60)

    # Looper over each numerical feature
    for (i,feature) in enumerate(numerical_features):

        # Calculate the proportion of missing value
        missing_ratio_i = df[feature].isnull().sum() / df.shape[0] * 100

        # Plot the graph of distribution with kde
        fig = sns.histplot(data=df,
                           x=feature,
                           stat='density',
                           kde=True,
                           ax=axes[i],
                           hue='classification',
                           legend=True)
        
        # Set the layout
        axes[i].set_title(f"{feature} | {missing_ratio_i:.2f}% missing values", fontsize=30)
        axes[i].set_xlabel(feature, fontsize=25)
        axes[i].set_ylabel('Probability density', fontsize=25)
        axes[i].tick_params(axis='x', labelsize=20)
        axes[i].tick_params(axis='y', labelsize=20)
        axes[i].legend(fontsize=20)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return figure, axes


def categorical_countplot(df, categorical_features, n_rows, n_cols):
    """
    Plot the countplot of multiple categorical features in a grid of subplots and the proportion of missing value
    and the proportion of missing value for each feature

    Inputs:
        df[pandas.DataFrame] : the dataframe containing the data
        categorical_features[list] : list of categorical feature names
        n_rows[Int] : number of rows in the subplot grid.
        n_cols[Int] : number of columns in the subplot grid.
    Outputs:
        figure, axes
    """
    # Create a figure with a grid of subplots
    figure, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(60,90))

    axes = axes.flatten()

    # Define the title of the figure
    figure.suptitle('Countplot of categorical features', fontsize=60)

    # Loop over each categorical feature
    for (i, feature) in enumerate(categorical_features):

        # Calculate the proportion of missing value
        missing_ratio_i = df[feature].isnull().sum() / df.shape[0] * 100

        # Define the legend
        label_i = f"{feature} | {missing_ratio_i:.2f}% missing values"

        # Plot the countplot
        fig = sns.countplot(x=df[feature],
                            label=label_i,
                            ax=axes[i])
        
        # Set the layout
        axes[i].set_xlabel(feature, fontsize=30)
        axes[i].set_ylabel('Counting', fontsize=30)
        axes[i].tick_params(axis='x', labelsize=25)
        axes[i].tick_params(axis='y', labelsize=25)
        axes[i].legend(fontsize=30)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return figure, axes


def plot_comparison(cleaned_df, preprocessed_df):
    '''
    Compare the distribution / countplot graphes of features with more than 10% missing values,
    between the cleaned dataframe and the preprocessed dataframe.

    Input:
        df[pandas.DataFrame] : the cleaned dataframe, before encoding and imputing.
        preprocessed[pandas.DataFrame] : the preprocessed dataframe
    Output:
        figure, axes
    '''

    # Retrieve features with more than 10% missing values
    missing_features = [col for col in cleaned_df.columns if (cleaned_df[col].isnull().sum()/cleaned_df.shape[0]) >= 0.1]

    numerical_missing_features = [col for col in missing_features if preprocessed_df[col].nunique() > 2]
    categorical_missing_features = [col for col in missing_features if preprocessed_df[col].nunique() <= 2]

    n_rows, n_cols = len(missing_features), 2

    figure, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(60,100))

    figure.suptitle("Comparaison before vs. after imputation for features with >= 10% missing values", fontsize=100)

    for (i,feature) in enumerate(missing_features):

        missing_ratio_i = cleaned_df[feature].isnull().sum() / cleaned_df.shape[0] * 100

        label_i = f"{missing_ratio_i:.2f}% missing values"

        if feature in numerical_missing_features:

            fig = sns.histplot(data = cleaned_df,
                               x=feature,
                               stat='density',
                               kde=True,
                               ax=axes[i,0],
                               hue='classification')
            axes[i,0].set_title(f"Before - {feature} | {label_i}", fontsize=35)
            axes[i,0].set_xlabel(feature, fontsize=30)
            axes[i,0].set_ylabel('Density', fontsize=30)
            axes[i,0].tick_params(axis='x', labelsize=25)
            axes[i,0].tick_params(axis='y', labelsize=25)
            axes[i,0].legend(fontsize=35)

            fig = sns.histplot(data = preprocessed_df,
                               x = feature,
                               stat='density',
                               kde=True,
                               ax=axes[i,1],
                               hue='classification')
            axes[i,1].set_title(f"After - {feature}", fontsize=35)
            axes[i,1].set_xlabel(feature, fontsize=30)
            axes[i,1].set_ylabel('Density', fontsize=30)
            axes[i,1].tick_params(axis='x', labelsize=25)
            axes[i,1].tick_params(axis='y', labelsize=25)
            axes[i,1].legend(fontsize=35)

        elif feature in categorical_missing_features:

            fig = sns.countplot(x=cleaned_df[feature],
                                ax=axes[i,0],
                                label=label_i)
            axes[i,0].set_title(f"Before - {feature}", fontsize=35)
            axes[i,0].set_xlabel(feature, fontsize=30)
            axes[i,0].set_ylabel('Counting', fontsize=30)
            axes[i,0].tick_params(axis='x', labelsize=25)
            axes[i,0].tick_params(axis='y', labelsize=25)
            axes[i,0].legend(fontsize=35)


            fig = sns.countplot(x=preprocessed_df[feature].map({0:'abnormal', 1:'normal'}),
                               ax=axes[i,1])
            axes[i,1].set_title(f"After - {feature}", fontsize=35)
            axes[i,1].set_xlabel(feature, fontsize=30)
            axes[i,1].set_ylabel('Counting', fontsize=30)
            axes[i,1].tick_params(axis='x', labelsize=25)
            axes[i,1].tick_params(axis='y', labelsize=25)
            axes[i,1].legend(fontsize=35)            

    plt.subplots_adjust(hspace=0.5, wspace=0.15)
    figure.tight_layout(rect=[0, 0, 1, 0.97])
    return figure, axes


def numerical_target(preprocessed_df):
    """
    Plot the violin graphes of multiple numerical features in a grid of subplots, comparing the distribution between both target classes.

    Inputs:
        preprocessed_df[pandas.DataFrame] : the dataframe containing the preprocessed data
    Outputs:
        figure, axes
    """
    copy_df = preprocessed_df.copy()
    copy_df['Label'] = preprocessed_df['classification'].map({1:'ckd', 0:'not ckd'})

    numerical_features = [col for col in preprocessed_df.columns if preprocessed_df[col].nunique() > 2]
    
    n_cols = 3
    n_rows = len(numerical_features) // n_cols + 1

    figure, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(40,50))
    axes = axes.flatten()
    figure.suptitle('Numerical feature distributions for both target classes', fontsize=60)

    for (i, feature) in enumerate(numerical_features):

        fig = sns.violinplot(data = copy_df,
                            x = 'Label',
                            y = feature,
                            ax=axes[i],
                            hue = 'Label',
                            palette='Set1')
        axes[i].set_title(f'{feature} distribution', fontsize=35)
        axes[i].set_xlabel('Label class', fontsize=25)
        axes[i].set_ylabel(feature, fontsize=25)
        axes[i].tick_params(axis='x', labelsize=20)
        axes[i].tick_params(axis='y', labelsize=20)
        axes[i].legend(fontsize=25)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return figure, axes