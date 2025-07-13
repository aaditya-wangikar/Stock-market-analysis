import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from statsmodels.tsa.arima.model import ARIMA
from datetime import timedelta
import seaborn as sns
import warnings
import pandas as pd
import pdb
warnings.filterwarnings('ignore')

from stocks import sector_stocks
from stock_prediction import display_top_stocks, moving_average, support_resistance, RSI, intraday, predict_future_values
from portfolio_optimiser import optimise_portfolio

analyze_stocks = input("""
                       Analyze stocks for:
                           1. Top performing stocks overall
                           2. Top performing stocks particular sector
                           3. User specific stocks
                       """)
period = input("Enter period to analyze all the stocks (e.g., 14d, 1mo, 6mo, 1y): ").strip()

trading_stratergy = input("""What is your trading stratergy?
                          1. Long term
                          2. Intraday
                          """)

# =============================================================================
# Identify top stocks
# =============================================================================
# List of NSE stock tickers
if analyze_stocks == '1':
    tickers_all = sector_stocks(0, return_all = True)
    display_top_stocks(tickers_all, period)
   

    # =============================================================================
    # Optimise portfolio for all the sectors
    # =============================================================================
    allocation_df, total_invested, remaining_cash = optimise_portfolio(tickers_all,ticker_type = 'all',period = period)

# =============================================================================
# Prompt for sector
# =============================================================================
elif analyze_stocks == '2':
    sectors = ['NIFTY50', 'AI', 'DEFENSE', 'PHARMA', 'ENERGY', 'FMCG', 'BANKING', 'INFRA', 'METALS', 'GOLD ETF','AUTO', 'CHEMICALS']
    print("Choose a sector:")
    for i, prompt in enumerate(sectors, start=1):
        print(f"{i}. {prompt}")
    sector_choice = int(input(f"Enter your sector choice (1-{len(sectors)}): "))    
    if 1 <= sector_choice <= len(sectors):
        selected_prompt = sectors[sector_choice - 1]
        print(f"\nYou selected: {selected_prompt} Stocks")
    else:
        print("Invalid choice!")
    
    sel_sect_tickers = sector_stocks(sector_choice)
    display_top_stocks(sel_sect_tickers, period=period)
    # =============================================================================
    # Optimise portfolio for given sector
    # =============================================================================
    allocation_df_sect, total_invested_sect, remaining_cash_sect = optimise_portfolio(sel_sect_tickers,ticker_type='sector',period = period)

    # =============================================================================
    # Prompt for a particular stock in a sector
    # =============================================================================
    print("Choose a sector:")
    for i, prompt in enumerate(sel_sect_tickers, start=1):
        print(f"{i}. {prompt.split('.')[0]}")
    stock_choice = int(input(f"Enter your stock choice (1-{len(sel_sect_tickers)}): "))    
    if 1 <= stock_choice <= len(sel_sect_tickers):
        selected_prompt = sel_sect_tickers[stock_choice - 1]
        print(f"\nYou selected: {selected_prompt.split('.')[0]}")
    else:
        print("Invalid choice!")
    tickers = [selected_prompt]

elif analyze_stocks == '3': 
    # =============================================================================
    # Optimise portfolio for user specific stocks
    # =============================================================================
    user_stocks = ['HAL.NS','MAZDOCK.NS','COCHINSHIP.NS','IDEAFORGE.NS','PARAS.NS','M&M.NS']
    allocation_df_user, total_invested_user, remaining_cash_user = optimise_portfolio(user_stocks,ticker_type='user',period = period)

    for i, prompt in enumerate(user_stocks, start=1):
        print(f"{i}. {prompt.split('.')[0]}")
    stock_choice = int(input(f"Enter your stock choice (1-{len(user_stocks)}): "))    
    if 1 <= stock_choice <= len(user_stocks):
        selected_prompt = user_stocks[stock_choice - 1]
        print(f"\nYou selected: {selected_prompt.split('.')[0]}")
    else:
        print("Invalid choice!")
    tickers = [selected_prompt]


# =============================================================================
# Extract data  
# =============================================================================
if trading_stratergy == '1':
    moving_average(tickers)
    support_resistance(tickers)
    RSI(tickers)
elif trading_stratergy == '2':
    intraday(tickers)

# =============================================================================
# Predict values for next 7 days
# =============================================================================
print(f'\n >> Predicted values of {tickers} for next 7 days: \n')
predict_future_values(tickers)