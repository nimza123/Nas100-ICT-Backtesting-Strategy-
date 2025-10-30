# NAS100 ICT Strategy Backtester

A deterministic backtesting system for NAS100 using ICT (Inner Circle Trader) concepts including FVG (Fair Value Gaps) and CISD patterns.

## 📊 Strategy Overview

- Timeframes: 1H + 5M multi-timeframe analysis
- Patterns: FVG (Fair Value Gaps), Bullish FVG, CISD (Change in Supply/Demand)
- Risk Management: Position sizing, daily trade limits, risk per trade
- Data: Live NAS100 data from Yahoo Finance

## 🚀 Features

- Deterministic backtesting (reproducible results)
- Live market data integration
- Comprehensive performance analytics
- Equity curve visualization
- Day-of-week performance analysis
- Risk management controls

## 📈 Performance Metrics

- Final balance & total return
- Win rate & consecutive wins/losses
- Day-of-week performance breakdown
- Maximum drawdown
- Trade-by-trade analysis

## 🛠 Installation

```bash
git clone https://github.com/nimza123/nas100-ict-backtester.git
cd nas100-ict-backtester
pip install -r requirements.txt

python nas100_ict_strategy.py

Follow the interactive prompts to:

1.Set account size and risk parameters

2.Choose backtest period

3.Run strategy analysis

4.View performance reports and charts

📊 PERFORMANCE REPORT
======================================================================
💰 CAPITAL & RETURNS:
   Starting Balance: $10,000.00
   Final Balance: $11,250.00
   Total Return: +12.50%

🎯 TRADE METRICS:
   Total Trades: 24
   Winning Trades: 16
   Losing Trades: 8
   Win Rate: 66.7%
   Max Consecutive Wins: 5
   Max Consecutive Losses: 3

   See requirements.txt for all dependencies.