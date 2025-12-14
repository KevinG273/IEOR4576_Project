"""
Utility functions module: unified tools for machine learning asset pricing prediction
Includes data loading, evaluation metrics, portfolio construction, and other functions
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy import stats
from scipy.optimize import minimize
from sklearn.decomposition import PCA

# Optional progress bar
try:
    from tqdm import tqdm
except ImportError:
    # If tqdm not available, use identity function
    def tqdm(iterable, desc=None, leave=None):
        return iterable


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


def calculate_rolling_covariance_empirical(returns_df, window=12, min_periods=6, target_dates=None):
    """
    Calculate rolling covariance matrix using empirical method (optimized)
    
    Parameters
    ----------
    returns_df : pd.DataFrame
        DataFrame with columns: date, id, return
    window : int
        Rolling window size in months (default 12)
    min_periods : int
        Minimum periods required for calculation (default 6)
    target_dates : list, optional
        Only calculate covariance for these dates (for efficiency)
    
    Returns
    -------
    cov_dict : dict
        Dictionary with date as key and covariance matrix as value
    """
    # Pivot to get returns matrix (dates x stocks)
    returns_pivot = returns_df.pivot_table(
        index='date', 
        columns='id', 
        values='return', 
        aggfunc='mean'
    )
    
    # Sort by date
    returns_pivot = returns_pivot.sort_index()
    
    # If target_dates specified, only calculate for those dates
    if target_dates is not None:
        dates_to_process = sorted(set(target_dates) & set(returns_pivot.index))
    else:
        dates_to_process = returns_pivot.index.tolist()
    
    cov_dict = {}
    
    # Use vectorized rolling window calculation
    for date in tqdm(dates_to_process, desc="Calculating empirical covariance", leave=False):
        # Find the position of this date
        date_idx = returns_pivot.index.get_loc(date)
        
        # Get rolling window data
        start_idx = max(0, date_idx - window + 1)
        date_data = returns_pivot.iloc[start_idx:date_idx+1]
        
        if len(date_data) < min_periods:
            continue
        
        # Remove stocks with insufficient data (vectorized)
        valid_stocks = date_data.notna().sum(axis=0) >= min_periods
        date_data = date_data.loc[:, valid_stocks]
        
        if date_data.shape[1] < 2:
            continue
        
        # Calculate empirical covariance (much faster than per-date loop)
        cov_matrix = date_data.cov()
        
        # Regularize (add small diagonal for numerical stability)
        cov_matrix = cov_matrix + np.eye(cov_matrix.shape[0]) * 1e-6
        
        cov_dict[date] = cov_matrix
    
    return cov_dict


def calculate_rolling_covariance_pca(returns_df, window=12, min_periods=6, n_components=None, 
                                     explained_variance=0.90, target_dates=None):
    """
    Calculate rolling covariance matrix using PCA-based method (optimized)
    
    Parameters
    ----------
    returns_df : pd.DataFrame
        DataFrame with columns: date, id, return
    window : int
        Rolling window size in months (default 12)
    min_periods : int
        Minimum periods required for calculation (default 6)
    n_components : int, optional
        Number of principal components to use. If None, use explained_variance
        Fixed n_components is much faster than calculating for each date
    explained_variance : float
        Target explained variance ratio (default 0.90, reduced for speed)
    
    Returns
    -------
    cov_dict : dict
        Dictionary with date as key and covariance matrix as value
    """
    # Pivot to get returns matrix
    returns_pivot = returns_df.pivot_table(
        index='date', 
        columns='id', 
        values='return', 
        aggfunc='mean'
    )
    
    # Sort by date
    returns_pivot = returns_pivot.sort_index()
    
    # If target_dates specified, only calculate for those dates
    if target_dates is not None:
        dates_to_process = sorted(set(target_dates) & set(returns_pivot.index))
    else:
        dates_to_process = returns_pivot.index.tolist()
    
    # Pre-determine n_components if not specified (use first valid window)
    if n_components is None:
        # Find first date with enough data
        for date in dates_to_process[:10]:  # Check first few dates
            date_idx = returns_pivot.index.get_loc(date)
            start_idx = max(0, date_idx - window + 1)
            date_data = returns_pivot.iloc[start_idx:date_idx+1]
            valid_stocks = date_data.notna().sum(axis=0) >= min_periods
            date_data = date_data.loc[:, valid_stocks]
            
            if date_data.shape[1] >= 2:
                # Determine n_components once
                date_data_filled = date_data.fillna(date_data.mean())
                mean_returns = date_data_filled.mean(axis=0)
                std_returns = date_data_filled.std(axis=0).replace(0, 1)
                date_data_std = (date_data_filled - mean_returns) / std_returns
                
                pca_temp = PCA()
                pca_temp.fit(date_data_std)
                cumsum_var = np.cumsum(pca_temp.explained_variance_ratio_)
                n_components = np.argmax(cumsum_var >= explained_variance) + 1
                n_components = min(n_components, max(5, date_data_std.shape[1] // 2))  # Cap at reasonable size
                break
    
    cov_dict = {}
    
    for date in tqdm(dates_to_process, desc="Calculating PCA covariance", leave=False):
        # Find the position of this date
        date_idx = returns_pivot.index.get_loc(date)
        
        # Get rolling window data
        start_idx = max(0, date_idx - window + 1)
        date_data = returns_pivot.iloc[start_idx:date_idx+1]
        
        if len(date_data) < min_periods:
            continue
        
        # Remove stocks with insufficient data
        valid_stocks = date_data.notna().sum(axis=0) >= min_periods
        date_data = date_data.loc[:, valid_stocks]
        
        if date_data.shape[1] < 2:
            continue
        
        # Fill remaining NaN with column mean (vectorized)
        date_data = date_data.fillna(date_data.mean())
        
        # Standardize (vectorized)
        mean_returns = date_data.mean(axis=0)
        std_returns = date_data.std(axis=0)
        std_returns = std_returns.replace(0, 1)
        date_data_std = (date_data - mean_returns) / std_returns
        
        # Apply PCA with fixed n_components (much faster)
        n_comp = min(n_components or 10, date_data_std.shape[1] - 1, date_data_std.shape[0] - 1)
        if n_comp < 1:
            continue
            
        pca = PCA(n_components=n_comp)
        pca.fit(date_data_std)
        
        # Reconstruct covariance matrix from PCA
        loadings = pca.components_.T
        eigenvalues = pca.explained_variance_
        
        # Reconstruct covariance (vectorized)
        cov_matrix = loadings @ np.diag(eigenvalues) @ loadings.T
        
        # Scale back to original units
        cov_matrix = np.outer(std_returns, std_returns) * cov_matrix
        
        # Regularize
        cov_matrix = cov_matrix + np.eye(cov_matrix.shape[0]) * 1e-6
        
        # Convert to DataFrame
        cov_matrix = pd.DataFrame(
            cov_matrix, 
            index=date_data.columns, 
            columns=date_data.columns
        )
        
        cov_dict[date] = cov_matrix
    
    return cov_dict


def mean_variance_optimize(expected_returns, cov_matrix, risk_aversion=1.0, 
                          long_only=True, max_weight=0.1, min_weight=0.0):
    """
    Mean-variance portfolio optimization (optimized for speed)
    
    Objective: maximize w^T * mu - lambda * w^T * Sigma * w
    where lambda is risk aversion parameter
    
    Parameters
    ----------
    expected_returns : np.array or pd.Series
        Expected returns for each stock
    cov_matrix : np.array or pd.DataFrame
        Covariance matrix
    risk_aversion : float
        Risk aversion parameter (higher = more risk averse)
    long_only : bool
        If True, only allow long positions (weights >= 0)
    max_weight : float
        Maximum weight per stock (default 0.1 = 10%)
    min_weight : float
        Minimum weight per stock (default 0.0)
    
    Returns
    -------
    weights : np.array
        Optimal portfolio weights
    """
    n = len(expected_returns)
    
    # Convert to numpy arrays
    if isinstance(expected_returns, pd.Series):
        expected_returns = expected_returns.values
    if isinstance(cov_matrix, pd.DataFrame):
        cov_matrix = cov_matrix.values
    
    # For small portfolios, use analytical solution if possible
    if n <= 3 and long_only and max_weight >= 1.0 / n:
        # Simple equal weights for very small portfolios
        return np.ones(n) / n
    
    # Objective function: minimize negative utility
    # - (w^T * mu - lambda * w^T * Sigma * w)
    def objective(w):
        portfolio_return = np.dot(w, expected_returns)
        portfolio_variance = np.dot(w, np.dot(cov_matrix, w))
        return -(portfolio_return - risk_aversion * portfolio_variance)
    
    # Constraints: weights sum to 1
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    
    # Bounds
    if long_only:
        bounds = [(min_weight, max_weight) for _ in range(n)]
    else:
        bounds = [(-max_weight, max_weight) for _ in range(n)]
    
    # Initial guess: equal weights
    x0 = np.ones(n) / n
    
    # Optimize with reduced iterations for speed
    try:
        result = minimize(
            objective, 
            x0, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-6}  # Reduced iterations
        )
        
        if result.success:
            weights = result.x
            # Normalize to ensure sum = 1
            weights = weights / np.sum(weights)
            return weights
        else:
            # Fallback to equal weights if optimization fails
            return np.ones(n) / n
    except:
        # Fallback to equal weights if optimization fails
        return np.ones(n) / n


def build_portfolio_returns_optimized(y_true, y_pred, dates, ids, 
                                     historical_returns_df=None,
                                     method='empirical',  # 'empirical' or 'pca'
                                     window=12,
                                     risk_aversion=1.0,
                                     long_only=True,
                                     max_weight=0.1,
                                     top_n=None,  # If specified, only use top N stocks by prediction
                                     n_deciles=10):
    """
    Build optimized portfolio returns using rolling covariance and mean-variance optimization
    
    For each month:
    1. Select stocks (top N by prediction or by decile)
    2. Calculate rolling covariance matrix (empirical or PCA-based)
    3. Use mean-variance optimizer to find optimal weights
    4. Calculate portfolio return
    
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
    historical_returns_df : pd.DataFrame, optional
        Historical returns for covariance calculation. 
        Must have columns: date, id, return
        If None, will use y_true (limited historical data)
    method : str
        'empirical' or 'pca' for covariance calculation
    window : int
        Rolling window size for covariance (default 12 months)
    risk_aversion : float
        Risk aversion parameter (default 1.0)
    long_only : bool
        If True, only long positions (default True)
    max_weight : float
        Maximum weight per stock (default 0.1 = 10%)
    top_n : int, optional
        If specified, only use top N stocks by prediction
    n_deciles : int
        Number of deciles for long-short strategy (default 10)
    
    Returns
    -------
    portfolio_returns : pd.DataFrame
        Contains monthly portfolio returns
    summary_stats : dict
        Summary statistics
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
    
    # Prepare historical returns for covariance calculation
    if historical_returns_df is None:
        # Use current data as historical (limited)
        historical_returns = df[['date', 'id', 'y_true']].copy()
        historical_returns = historical_returns.rename(columns={'y_true': 'return'})
    else:
        historical_returns = historical_returns_df.copy()
    
    # Get unique dates for which we need covariance matrices
    dates_sorted = sorted(df['date'].unique())
    
    # Calculate rolling covariance (only for dates we actually need)
    print(f"Calculating rolling covariance using {method} method...")
    if method == 'pca':
        cov_dict = calculate_rolling_covariance_pca(
            historical_returns, 
            window=window, 
            target_dates=dates_sorted
        )
    else:
        cov_dict = calculate_rolling_covariance_empirical(
            historical_returns, 
            window=window, 
            target_dates=dates_sorted
        )
    
    # Group by date
    portfolio_returns = []
    
    for date in tqdm(dates_sorted, desc="Building optimized portfolio", leave=False):
        date_data = df[df['date'] == date].copy()
        
        if len(date_data) < 2:
            continue
        
        # Sort by predicted values
        date_data = date_data.sort_values('y_pred', ascending=False)
        
        # Calculate deciles once (avoid duplicate calculation)
        if top_n is None:
            date_data['decile'] = pd.qcut(
                date_data['y_pred'], 
                q=n_deciles, 
                labels=False, 
                duplicates='drop'
            ) + 1
        
        # Select stocks for long portfolio
        if top_n is not None:
            long_stocks = date_data.head(top_n).copy()
        else:
            # Use top decile
            long_stocks = date_data[date_data['decile'] == n_deciles].copy()
        
        # Select stocks for short portfolio (bottom decile)
        if top_n is None:
            short_stocks = date_data[date_data['decile'] == 1].copy()
        else:
            # For top_n case, use bottom N
            short_stocks = date_data.tail(top_n).copy()
        
        if len(long_stocks) == 0 or len(short_stocks) == 0:
            portfolio_returns.append({
                'date': date,
                'long_short': np.nan,
                'long_return': np.nan,
                'short_return': np.nan
            })
            continue
        
        # Initialize return variables
        long_return = np.nan
        short_return = np.nan
        ls_return = np.nan
        
        # Get covariance matrix for this date (use most recent available)
        available_dates = [d for d in cov_dict.keys() if d <= date]
        if len(available_dates) == 0:
            # Fallback to equal weights
            long_return = long_stocks['y_true'].mean()
            short_return = short_stocks['y_true'].mean()
            ls_return = long_return - short_return
        else:
            cov_date = max(available_dates)
            cov_matrix = cov_dict[cov_date]
            
            # Get common stocks between portfolio and covariance matrix
            long_ids = set(long_stocks['id'].values)
            short_ids = set(short_stocks['id'].values)
            cov_stocks = set(cov_matrix.index)
            
            long_common = list(long_ids & cov_stocks)
            short_common = list(short_ids & cov_stocks)
            
            if len(long_common) == 0 or len(short_common) == 0:
                # Fallback to equal weights
                long_return = long_stocks['y_true'].mean()
                short_return = short_stocks['y_true'].mean()
                ls_return = long_return - short_return
            else:
                # Long portfolio optimization
                long_pred = long_stocks[long_stocks['id'].isin(long_common)]['y_pred'].values
                long_cov = cov_matrix.loc[long_common, long_common]
                
                try:
                    long_weights = mean_variance_optimize(
                        long_pred, 
                        long_cov,
                        risk_aversion=risk_aversion,
                        long_only=long_only,
                        max_weight=max_weight
                    )
                    long_actual = long_stocks[long_stocks['id'].isin(long_common)]['y_true'].values
                    long_return = np.dot(long_weights, long_actual)
                except:
                    long_return = long_stocks['y_true'].mean()
                
                # Short portfolio optimization (inverse weights)
                short_pred = short_stocks[short_stocks['id'].isin(short_common)]['y_pred'].values
                short_cov = cov_matrix.loc[short_common, short_common]
                
                try:
                    # For short, we want to minimize expected return (inverse prediction)
                    short_pred_inv = -short_pred
                    short_weights = mean_variance_optimize(
                        short_pred_inv,
                        short_cov,
                        risk_aversion=risk_aversion,
                        long_only=long_only,
                        max_weight=max_weight
                    )
                    short_actual = short_stocks[short_stocks['id'].isin(short_common)]['y_true'].values
                    short_return = np.dot(short_weights, short_actual)
                except:
                    short_return = short_stocks['y_true'].mean()
                
                ls_return = long_return - short_return
        
        portfolio_returns.append({
            'date': date,
            'long_short': ls_return,
            'long_return': long_return,
            'short_return': short_return
        })
    
    portfolio_df = pd.DataFrame(portfolio_returns)
    portfolio_df = portfolio_df.sort_values('date').reset_index(drop=True)
    
    # Calculate summary statistics
    if len(portfolio_df) > 0 and 'long_short' in portfolio_df.columns:
        ls_returns = portfolio_df['long_short'].dropna()
        
        if len(ls_returns) > 0:
            annual_return = ls_returns.mean() * 12
            annual_vol = ls_returns.std() * np.sqrt(12)
            sharpe = annual_return / annual_vol if annual_vol > 0 else np.nan
            cumulative_return = (1 + ls_returns).prod() - 1
            
            summary_stats = {
                'annual_return': annual_return,
                'annual_volatility': annual_vol,
                'sharpe_ratio': sharpe,
                'cumulative_return': cumulative_return,
                'n_months': len(ls_returns),
                'mean_monthly_return': ls_returns.mean(),
                'std_monthly_return': ls_returns.std(),
                'method': method,
                'risk_aversion': risk_aversion
            }
        else:
            summary_stats = {}
    else:
        summary_stats = {}
    
    return portfolio_df, summary_stats

