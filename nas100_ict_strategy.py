import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

plt.rcParams['axes.grid'] = True

# ===================== RISK MANAGER =====================
class RiskManager:
    def __init__(self):
        self.account_size = 10000.0
        self.risk_percent = 1.0
        self.trades_today = 0
        self.max_trades_per_day = 2

    def get_user_inputs(self):
        print("\n=== USER SETUP ===")
        self.account_size = float(input("Enter account balance ($): "))
        self.risk_percent = float(input("Enter risk per trade (%): "))
        print(f"\nAccount Size: ${self.account_size:,.2f}")
        print(f"Risk Per Trade: {self.risk_percent:.2f}%\n")

    def calculate_position_size(self, entry_price, stop_loss_price):
        risk_amount = self.account_size * (self.risk_percent / 100)
        stop_distance = abs(entry_price - stop_loss_price)
        if stop_distance == 0:
            return 0, 0, 0
        lot_size = risk_amount / stop_distance
        position_value = lot_size * entry_price
        return lot_size, position_value, risk_amount

    def can_take_trade(self):
        if self.trades_today >= self.max_trades_per_day:
            return False
        return True

    def record_trade(self):
        self.trades_today += 1


# ===================== DATE HANDLER =====================
class NAS100DateHandler:
    def get_backtest_period(self):
        print("\n=== DATE SETUP ===")
        today = datetime.now().date()
        default_start = today - timedelta(days=30)
        print(f"Default: {default_start} → {today}")
        user_input = input("Enter start and end date (YYYY-MM-DD YYYY-MM-DD) or press Enter for default: ").strip()
        if user_input:
            try:
                s, e = user_input.split()
                return datetime.fromisoformat(s), datetime.fromisoformat(e)
            except:
                print("Invalid input, using default 30 days.")
        return default_start, today


# ===================== DATA FETCHER =====================
class NAS100DataFetcher:
    def __init__(self):
        self.symbols = ["NAS100USD", "^NDX", "QQQ"]

    def fetch_data(self, start_date, end_date):
        print("\n" + "="*60)
        print("📡 FETCHING MARKET DATA")
        print("="*60)

        start_dt = datetime.combine(start_date, datetime.min.time())
        end_dt = datetime.combine(end_date, datetime.min.time()) + timedelta(days=1)
        df_1h = df_5m = None
        used_symbol = None

        for symbol in self.symbols:
            print(f"Trying symbol: {symbol}...")
            try:
                df_1h = yf.download(symbol, start=start_dt, end=end_dt, interval="1h", progress=False)
                df_5m = yf.download(symbol, start=start_dt, end=end_dt, interval="5m", progress=False)
                if not df_1h.empty and not df_5m.empty:
                    used_symbol = symbol
                    print(f"✅ Using symbol: {symbol}")
                    break
            except Exception as e:
                print(f"❌ Failed for {symbol}: {e}")

        if df_1h is None or df_1h.empty:
            print("❌ Could not fetch valid data.")
            return None, None, None

        # --- Normalize columns ---
        def normalize(df):
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            df.columns = [c.capitalize() for c in df.columns]
            return df

        df_1h = normalize(df_1h)
        df_5m = normalize(df_5m)

        # --- Validate OHLC columns ---
        required = {"Open", "High", "Low", "Close"}
        if not required.issubset(df_1h.columns):
            print(f"❌ Missing OHLC columns: {df_1h.columns}")
            return None, None, None

        for df in [df_1h, df_5m]:
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            df.index = df.index.tz_convert("America/New_York")
            for col in required:
                df[col] = df[col].round(5)

        df_daily = df_1h.resample("1D").agg({
            "Open": "first", "High": "max", "Low": "min", "Close": "last"
        }).dropna()

        print(f"\n📊 Data Summary for {used_symbol}:")
        print(f"   {len(df_daily)} daily candles")
        print(f"   {len(df_1h)} 1H candles")
        print(f"   {len(df_5m)} 5M candles\n")

        return df_daily, df_1h, df_5m


# ===================== STRATEGY =====================
class NAS100ICTStrategy:
    def detect_fvg(self, df):
        df["fvg"] = ((df["Low"].shift(-2) > df["High"]) & (df["Close"] > df["Open"]))
        return df

    def detect_bullish_fvg(self, df):
        df["bullish_fvg"] = (
            (df["Close"] > df["Open"]) &
            (df["Close"].shift(1) > df["Open"].shift(1)) &
            (df["Close"].shift(2) > df["Open"].shift(2)) &
            (df["Low"].shift(2) > df["High"])
        )
        return df

    def detect_cisd(self, df):
        df["cisd"] = (df["Close"] > df["Open"]) & (df["Low"] > df["Low"].shift(1))
        return df

    def generate_signals(self, df_1h, df_5m):
        df_1h = self.detect_fvg(self.detect_bullish_fvg(df_1h))
        df_5m = self.detect_fvg(self.detect_cisd(df_5m))

        signals = []
        for ts, row in df_5m.iterrows():
            if row["fvg"] or row["cisd"]:
                last_hour = df_1h[df_1h.index <= ts].iloc[-1]
                if last_hour["fvg"] or last_hour["bullish_fvg"]:
                    signals.append({
                        "timestamp": ts,
                        "entry_price": row["Close"],
                        "stop_loss": row["Low"],
                        "target": row["High"] * 1.02  # fixed RR 2:1 deterministic target
                    })
        return signals


# ===================== PERFORMANCE ANALYZER =====================
class NAS100PerformanceAnalyzer:
    def __init__(self, starting_balance):
        self.starting_balance = starting_balance
        self.balance = starting_balance
        self.equity_curve = []
        self.trades = []

    def record_trade(self, pnl, sig):
        self.balance += pnl
        self.equity_curve.append(self.balance)
        self.trades.append({
            "Time": sig["timestamp"],
            "Entry": sig["entry_price"],
            "SL": sig["stop_loss"],
            "TP": sig["target"],
            "PnL": pnl,
            "Win": pnl > 0
        })

    def calculate_consecutive_wins_losses(self):
        if not self.trades:
            return 0, 0
        
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_wins = 0
        current_losses = 0
        
        for trade in self.trades:
            if trade["PnL"] > 0:  # Win
                current_wins += 1
                current_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, current_wins)
            elif trade["PnL"] < 0:  # Loss
                current_losses += 1
                current_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, current_losses)
            else:  # Breakeven
                current_wins = 0
                current_losses = 0
                
        return max_consecutive_wins, max_consecutive_losses

    def calculate_day_of_week_performance(self):
        day_performance = {
            'Monday': {'trades': 0, 'pnl': 0, 'wins': 0},
            'Tuesday': {'trades': 0, 'pnl': 0, 'wins': 0},
            'Wednesday': {'trades': 0, 'pnl': 0, 'wins': 0},
            'Thursday': {'trades': 0, 'pnl': 0, 'wins': 0},
            'Friday': {'trades': 0, 'pnl': 0, 'wins': 0}
        }
        
        day_map = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday'}
        
        for trade in self.trades:
            day_num = trade['Time'].weekday()
            if day_num in day_map:
                day_name = day_map[day_num]
                day_performance[day_name]['trades'] += 1
                day_performance[day_name]['pnl'] += trade['PnL']
                if trade['PnL'] > 0:
                    day_performance[day_name]['wins'] += 1
        
        # Calculate win rates and returns
        for day in day_performance:
            if day_performance[day]['trades'] > 0:
                day_performance[day]['win_rate'] = (day_performance[day]['wins'] / day_performance[day]['trades']) * 100
                day_performance[day]['return_percent'] = (day_performance[day]['pnl'] / self.starting_balance) * 100
            else:
                day_performance[day]['win_rate'] = 0
                day_performance[day]['return_percent'] = 0
        
        return day_performance

    def report(self):
        if not self.trades:
            return "No trades executed."
        
        # Basic metrics
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t["PnL"] > 0])
        losing_trades = len([t for t in self.trades if t["PnL"] < 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        total_return = (self.balance - self.starting_balance) / self.starting_balance * 100
        
        # Consecutive wins/losses
        max_consecutive_wins, max_consecutive_losses = self.calculate_consecutive_wins_losses()
        
        # Day of week performance
        day_performance = self.calculate_day_of_week_performance()
        
        # Generate report
        report = "\n" + "="*70
        report += f"\n📊 PERFORMANCE REPORT"
        report += "\n" + "="*70
        
        report += f"\n💰 CAPITAL & RETURNS:"
        report += f"\n   Starting Balance: ${self.starting_balance:,.2f}"
        report += f"\n   Final Balance: ${self.balance:,.2f}"
        report += f"\n   Total Return: {total_return:+.2f}%"
        
        report += f"\n\n🎯 TRADE METRICS:"
        report += f"\n   Total Trades: {total_trades}"
        report += f"\n   Winning Trades: {winning_trades}"
        report += f"\n   Losing Trades: {losing_trades}"
        report += f"\n   Win Rate: {win_rate:.1f}%"
        report += f"\n   Max Consecutive Wins: {max_consecutive_wins}"
        report += f"\n   Max Consecutive Losses: {max_consecutive_losses}"
        
        report += f"\n\n📅 DAY OF WEEK PERFORMANCE:"
        for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
            perf = day_performance[day]
            if perf['trades'] > 0:
                report += f"\n   {day}: {perf['trades']:2d} trades, {perf['win_rate']:5.1f}% win rate, {perf['return_percent']:+.2f}% return"
            else:
                report += f"\n   {day}: No trades"
        
        report += "\n" + "="*70
        return report

    def show_trades(self):
        if self.trades:
            df = pd.DataFrame(self.trades)
            print("\nDetailed Trades:")
            print(df.to_string(index=False, justify='center', col_space=10))


# ===================== MAIN EXECUTION =====================
def main():
    print("\n" + "="*80)
    print("📊 NAS100 ICT STRATEGY — DETERMINISTIC BACKTEST (LIVE CFD DATA)")
    print("="*80)

    rm = RiskManager()
    rm.get_user_inputs()

    dh = NAS100DateHandler()
    start_date, end_date = dh.get_backtest_period()

    fetcher = NAS100DataFetcher()
    df_daily, df_1h, df_5m = fetcher.fetch_data(start_date, end_date)
    if df_1h is None or df_5m is None:
        print("❌ Unable to proceed — missing data.")
        return

    strat = NAS100ICTStrategy()
    analyzer = NAS100PerformanceAnalyzer(rm.account_size)
    signals = strat.generate_signals(df_1h, df_5m)

    print(f"\n🔍 Found {len(signals)} potential signals...")
    
    for sig in signals:
        if not rm.can_take_trade():
            continue

        entry, sl, tp = sig["entry_price"], sig["stop_loss"], sig["target"]
        lots, val, risk = rm.calculate_position_size(entry, sl)
        if lots <= 0:
            continue

        pnl = (tp - entry) * lots
        analyzer.record_trade(pnl, sig)
        rm.record_trade()

    print(analyzer.report())
    
    if analyzer.trades:
        show_details = input("\nShow detailed trades? (y/n): ").strip().lower()
        if show_details == 'y':
            analyzer.show_trades()

    if analyzer.equity_curve:
        plt.figure(figsize=(12, 8))
        
        # Equity curve
        plt.subplot(2, 1, 1)
        plt.plot(analyzer.equity_curve, linewidth=2, label="Equity Curve", color='blue')
        plt.axhline(y=analyzer.starting_balance, color='red', linestyle='--', alpha=0.7, label='Starting Balance')
        plt.title("NAS100 ICT Strategy — Equity Curve", fontsize=14, fontweight='bold')
        plt.xlabel("Trade #")
        plt.ylabel("Account Balance ($)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Day of week performance
        if analyzer.trades:
            plt.subplot(2, 1, 2)
            day_performance = analyzer.calculate_day_of_week_performance()
            days = list(day_performance.keys())
            win_rates = [day_performance[day].get('win_rate', 0) for day in days]
            
            colors = ['green' if rate > 50 else 'red' for rate in win_rates]
            bars = plt.bar(days, win_rates, color=colors, alpha=0.7)
            plt.title("Win Rate by Day of Week", fontsize=14, fontweight='bold')
            plt.ylabel("Win Rate (%)")
            plt.ylim(0, 100)
            
            # Add value labels on bars
            for bar, value in zip(bars, win_rates):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                        f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()