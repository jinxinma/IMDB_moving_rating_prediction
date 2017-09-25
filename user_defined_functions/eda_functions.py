import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np


def plot_dtype_distribution(df):
    """
    df: pandas dataframe

    this function plots a distribution of data types from the input dataframe
    """
    
    dataTypeDf = pd.DataFrame(df.dtypes.value_counts()).reset_index().rename(columns={'index':'Variable type', 0:'Count'})    
    dataTypeDf.plot(kind='bar', x='Variable type', y='Count', legend=None, title='Variables Count Across Datatype')


def movie_info_explore(df, col, top_n=10):
	"""
    df: pandas dataframe
    col: column name
    top_n: number of samples include in the output

	return: dataframe of movie title and  some numerical column (e.g. duration) 
            sorted by the value of the column in descending order
	"""
        # have to do this extra indentation otherwise can't import the script
        if df[col].dtype != 'O':
            return df[['movie_title', col]].sort_values(col, ascending=False)[:top_n]
        else: return -1


def get_missing_info(df):
    """
    df: pandas dataframe

    return: dataframe of count and percentage of missing value for 
            each column in the input dataframe
    """
    missing_ct = pd.DataFrame(df.isnull().sum()).reset_index().rename(columns={'index':'Feature Name', 0:'Count'})
    missing_ct['Percentage'] = (missing_ct['Count'] / len(df)) * 100
    return missing_ct.sort_values('Count', ascending=False)


def corr_mat_heatmap(df):
    """
    df: pandas dataframe

    this function plots a correlation matrix heatmap based on the input dataframe
    """
    corr_matrix = df.corr()
    
    sns.set(style='white', font_scale=1.4)
    f, ax = plt.subplots(figsize=(14, 9))
    
    # Create a mask for the upper triangle
    mask = np.zeros_like(corr_matrix, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    cmap = sns.diverging_palette(300, 50, as_cmap=True)
    sns.heatmap(corr_matrix, cmap=cmap, mask=mask, annot=True, annot_kws={'size': 12})  
    ax.set_title('Correlation Matrix Heatmap')


def top_frequent_value(df, col, top_n=10):
    """
    df: pandas dataframe
    col: column name

    return: dataframe of the frequency of attributes in a column
    """
    return  pd.DataFrame(df[col].value_counts()).reset_index().\
                         rename(columns={'index':col, col:'count'})[:top_n]

    
def subset_by_freq_value(df, col):
    """
    df: pandas dataframe
    col: column name
    
    return: dataframe of top n most frequent attributes in the input column
    """
    top_freq_col = top_frequent_value(df, col)[col]
    return df[df[col].isin(top_freq_col)]


def cat_boxplot(df, col):
    """
    df: pandas dataframe
    col: column name

    this function draws a box plot of imdb score grouped by attributes in categorical column
    """
    sns.set(style='whitegrid', font_scale=1.4)
    f, ax = plt.subplots(figsize=(10, 8))
    sns.boxplot(x=col, y='imdb_score', data=df) 
    ax.set_title('Boxplot Grouped by ' + col)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    
    
def cat_meanplot(df, col):
    """
    df: pandas dataframe
    col: column name

    this function draws a plot of average imdb_score grouped by attributes in a column
    e.g. mean imdb_score by director name
    """
    mean_df = pd.DataFrame(df.groupby([col])['imdb_score'].mean()).reset_index().rename(columns={'imdb_score':'mean_rating'})
    
    sns.set(style='whitegrid', font_scale=1.4)
    f, ax = plt.subplots(figsize=(10, 8))
    sns.pointplot(x=col, y='mean_rating', hue=col, data=mean_df, scale=2)
    
    ax.legend_.remove()
    ax.set_title('Mean Plot Grouped by ' + col)
    ax.set_ylabel('Mean IMDB Rating')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

