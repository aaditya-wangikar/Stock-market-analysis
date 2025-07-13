import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def optimise_portfolio(all_tickers, period, ticker_type):
    # ----------------- USER INPUTS -------------------
    
    budget = float(input("💰 Enter your total investment budget (e.g., 100000): "))

    # ----------------- FETCH PRICE DATA -------------------
    data = yf.download(all_tickers, period=period)['Close']
    data = data.dropna(axis=1, thresh=len(data) * 0.9)  # remove stocks with missing data
    returns = data.pct_change().dropna()
    
    # ----------------- SHARPE RATIO CALC -------------------
    annual_returns = returns.mean() * 252
    annual_volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = annual_returns / annual_volatility
    
    top5_tickers = sharpe_ratio.sort_values(ascending=False).head(5).index.tolist()
    print("\n📈 Top 5 Stocks by Sharpe Ratio:")
    print(top5_tickers)
    
    # ----------------- OPTIMIZATION -------------------
    top_returns = returns[top5_tickers]
    
    def portfolio_perf(weights):
        ret = np.sum(top_returns.mean() * weights) * 252
        vol = np.sqrt(np.dot(weights.T, np.dot(top_returns.cov() * 252, weights)))
        sharpe = ret / vol
        return ret, vol, sharpe
    
    def neg_sharpe(weights):
        return -portfolio_perf(weights)[2]
    
    num_assets = len(top5_tickers)
    bounds = tuple((0, 1) for _ in range(num_assets))
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    init_guess = [1/num_assets] * num_assets
    
    result = minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    weights = result.x
    
    # ----------------- ALLOCATION BASED ON BUDGET -------------------
    latest_prices = data[top5_tickers].iloc[-1]
    allocation_df = pd.DataFrame({
        'Ticker': top5_tickers,
        'Weight': weights,
        'Latest Price': latest_prices
    })
    
    allocation_df['Allocated Amount'] = allocation_df['Weight'] * budget
    allocation_df['Shares to Buy'] = (allocation_df['Allocated Amount'] / allocation_df['Latest Price']).astype(int)
    allocation_df['Actual Invested'] = allocation_df['Shares to Buy'] * allocation_df['Latest Price']
    if ticker_type == 'all':
        allocation_df.to_csv('Files//Optimised_portfolio_all_sectors.csv')
    elif ticker_type == 'sector':
        allocation_df.to_csv('Files//Optimised_portfolio_specific_sector.csv')
    elif ticker_type == 'user':
        allocation_df.to_csv('Files//Optimised_portfolio_specific_sector.csv')
    # ----------------- SUMMARY -------------------
    total_invested = allocation_df['Actual Invested'].sum()
    remaining_cash = budget - total_invested
    
    print("\n📊 Optimized Allocation:")
    print(allocation_df[['Ticker', 'Weight', 'Shares to Buy', 'Actual Invested']])
    
    print(f"\n✅ Total Invested: ₹{total_invested:.2f}")
    print(f"💰 Remaining Cash: ₹{remaining_cash:.2f}")
    
    # ----------------- PIE CHART -------------------
    plt.figure(figsize=(7, 6))
    plt.pie(allocation_df['Actual Invested'], labels=allocation_df['Ticker'],
            autopct='%1.1f%%', startangle=90)
    plt.title("Investment Allocation (Top 5 Sharpe Stocks)")
    plt.tight_layout()
    if ticker_type == 'all':
        plt.savefig('Files//Investment_Allocation_all_sectors.png')
    elif ticker_type == 'sector':
        plt.savefig('Files//Investment_Allocation__specific_sector.png')
    elif ticker_type == 'user':
        plt.savefig('Files//Investment_Allocation_user.png')
    plt.show()
    return allocation_df, total_invested, remaining_cash