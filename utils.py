"""
Utility functions module: unified tools for machine learning asset pricing prediction
Includes data loading, evaluation metrics, portfolio construction, and other functions
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy import stats


def load_data(filepath='ger_factor_data_from2003.csv'):
    """
    Load processed data
    
    Parameters
    ----------
    filepath : str
        Path to the data file
    
    Returns
    -------
    df : pd.DataFrame
        DataFrame containing features and target variables
    """
    df = pd.read_csv(filepath)
    df['eom'] = pd.to_datetime(df['eom'])
    df = df.sort_values(by=['id', 'eom']).reset_index(drop=True)
    return df


def prepare_features_target(df, exclude_cols=None):
    """
    Prepare feature matrix X and target variable y
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame
    exclude_cols : list
        Columns to exclude (e.g., id, eom, target variable, etc.)
    
    Returns
    -------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable (ret_exc_lead1m)
    metadata : pd.DataFrame
        Metadata (id, eom, etc.)
    feature_names : list
        List of feature names
    """
    if exclude_cols is None:
        exclude_cols = ['id', 'eom', 'ret_exc_lead1m', 'ret', 'ret_exc', 'year', 'month']
    
    # Get feature columns
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Extract features and target
    X = df[feature_cols].copy()
    y = df['ret_exc_lead1m'].copy()
    metadata = df[['id', 'eom']].copy()
    
    return X, y, metadata, feature_cols


def calculate_r2_os(y_true, y_pred):
    """
    Calculate Out-of-Sample R²
    
    R²_OS = 1 - (SSE_model / SSE_baseline)
    where baseline is the historical average return
    
    Parameters
    ----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    
    Returns
    -------
    r2_os : float
        Out-of-sample R²
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Remove NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        return np.nan
    
    # Calculate baseline (historical average)
    y_mean = np.mean(y_true)
    
    # SSE_model
    sse_model = np.sum((y_true - y_pred) ** 2)
    
    # SSE_baseline
    sse_baseline = np.sum((y_true - y_mean) ** 2)
    
    # R²_OS
    if sse_baseline == 0:
        return np.nan
    
    r2_os = 1 - (sse_model / sse_baseline)
    
    return r2_os


def build_portfolio_returns(y_true, y_pred, dates, ids, n_deciles=10):
    """
    Build portfolio returns based on prediction ranking
    
    For each month:
    1. Sort by predicted values
    2. Divide into n_deciles groups
    3. Calculate equal-weighted returns for each group
    4. Construct long-short portfolio (highest group - lowest group)
    
    Parameters
    ----------
    y_true : array-like
        True returns
    y_pred : array-like
        Predicted returns
    dates : array-like
        Dates
    ids : array-like
        Stock IDs
    n_deciles : int
        Number of groups (default 10)
    
    Returns
    -------
    portfolio_returns : pd.DataFrame
        Contains monthly portfolio returns and long-short returns
    summary_stats : dict
        Summary statistics (annualized return, volatility, Sharpe, etc.)
    """
    # Convert to DataFrame
    df = pd.DataFrame({
        'date': dates,
        'id': ids,
        'y_true': y_true,
        'y_pred': y_pred
    })
    
    # Remove NaN
    df = df.dropna(subset=['y_true', 'y_pred'])
    
    # Group by date
    portfolio_returns = []
    
    for date in df['date'].unique():
        date_data = df[df['date'] == date].copy()
        
        if len(date_data) < n_deciles:
            continue
        
        # Sort by predicted values
        date_data = date_data.sort_values('y_pred')
        
        # Divide into n_deciles groups
        date_data['decile'] = pd.qcut(
            date_data['y_pred'], 
            q=n_deciles, 
            labels=False, 
            duplicates='drop'
        ) + 1
        
        # Calculate equal-weighted returns for each group
        decile_returns = date_data.groupby('decile')['y_true'].mean()
        
        # Construct long-short portfolio (highest group - lowest group)
        if len(decile_returns) == n_deciles:
            ls_return = decile_returns.iloc[-1] - decile_returns.iloc[0]
        else:
            ls_return = np.nan
        
        # Save results
        result = {
            'date': date,
            'long_short': ls_return
        }
        
        # Add decile returns
        for d in range(1, n_deciles + 1):
            result[f'decile_{d}'] = decile_returns.get(d, np.nan)
        
        portfolio_returns.append(result)
    
    portfolio_df = pd.DataFrame(portfolio_returns)
    portfolio_df = portfolio_df.sort_values('date').reset_index(drop=True)
    
    # Calculate summary statistics
    if len(portfolio_df) > 0 and 'long_short' in portfolio_df.columns:
        ls_returns = portfolio_df['long_short'].dropna()
        
        if len(ls_returns) > 0:
            # Annualized return (assuming monthly data)
            annual_return = ls_returns.mean() * 12
            
            # Annualized volatility
            annual_vol = ls_returns.std() * np.sqrt(12)
            
            # Sharpe ratio (assuming risk-free rate is 0)
            sharpe = annual_return / annual_vol if annual_vol > 0 else np.nan
            
            # Cumulative return
            cumulative_return = (1 + ls_returns).prod() - 1
            
            summary_stats = {
                'annual_return': annual_return,
                'annual_volatility': annual_vol,
                'sharpe_ratio': sharpe,
                'cumulative_return': cumulative_return,
                'n_months': len(ls_returns),
                'mean_monthly_return': ls_returns.mean(),
                'std_monthly_return': ls_returns.std()
            }
        else:
            summary_stats = {}
    else:
        summary_stats = {}
    
    return portfolio_df, summary_stats


def calculate_prediction_metrics(y_true, y_pred):
    """
    Calculate prediction performance metrics
    
    Parameters
    ----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    
    Returns
    -------
    metrics : dict
        Dictionary containing various metrics
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Remove NaN
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        return {}
    
    metrics = {
        'r2_os': calculate_r2_os(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': np.mean(np.abs(y_true - y_pred)),
        'correlation': np.corrcoef(y_true, y_pred)[0, 1],
        'mean_pred': np.mean(y_pred),
        'mean_true': np.mean(y_true),
        'std_pred': np.std(y_pred),
        'std_true': np.std(y_true),
        'n_observations': len(y_true)
    }
    
    return metrics

