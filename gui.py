import tkinter as tk
from tkinter import ttk, messagebox
import tkinter.font as tkFont
import pandas as pd
import numpy as np
import requests
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
import time
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib style for dark theme
plt.style.use('dark_background')

class SimpleForexPredictor:
    """Simplified forex prediction model without TensorFlow dependency"""
    
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.window_size = 15
        self.weights = np.random.random(self.window_size)
        
    def prepare_data(self, data):
        """Prepare training data"""
        if len(data) < self.window_size:
            return None, None
            
        X, y = [], []
        for i in range(self.window_size, len(data)):
            X.append(data[i-self.window_size:i])
            y.append(data[i])
        return np.array(X), np.array(y)
    
    def train(self, price_data):
        """Train model (simplified version)"""
        try:
            # Normalize data
            scaled_data = self.scaler.fit_transform(price_data.reshape(-1, 1)).flatten()
            
            # Prepare training data
            X, y = self.prepare_data(scaled_data)
            if X is None:
                return False
                
            # Simple linear regression weight update
            for i in range(len(X)):
                prediction = np.dot(X[i], self.weights)
                error = y[i] - prediction
                self.weights += 0.001 * error * X[i]  # Simple gradient descent
                
            return True
        except Exception as e:
            print(f"Model training error: {e}")
            return False
    
    def predict(self, recent_data):
        """Predict next price"""
        try:
            if len(recent_data) < self.window_size:
                return recent_data[-1]  # Return last price
                
            # Normalize recent data
            scaled_recent = self.scaler.transform(recent_data[-self.window_size:].reshape(-1, 1)).flatten()
            
            # Predict
            prediction_scaled = np.dot(scaled_recent, self.weights)
            
            # Denormalize
            prediction = self.scaler.inverse_transform([[prediction_scaled]])[0][0]
            
            return prediction
        except Exception as e:
            print(f"Prediction error: {e}")
            return recent_data[-1] if len(recent_data) > 0 else 0

class ForexTradingGUI:
    def __init__(self, root):
        self.root = root
        self.setup_main_window()
        self.initialize_variables()
        self.initialize_predictors()
        self.create_widgets()
        self.setup_trading_environment()
        self.start_real_time_updates()
        
    def setup_main_window(self):
        """Setup main window"""
        self.root.title("🌍 2025 ML Final FX-Change®")
        self.root.geometry("1600x1000")
        self.root.configure(bg='#1e1e1e')
        
        # Setup fonts
        self.title_font = tkFont.Font(family="Arial", size=16, weight="bold")
        self.header_font = tkFont.Font(family="Arial", size=12, weight="bold")
        self.normal_font = tkFont.Font(family="Arial", size=12)
        
    def initialize_variables(self):
        """Initialize variables"""
        self.currency_pairs = ['USD/JPY', 'USD/EUR', 'USD/GBP']
        self.current_prices = {'USD/JPY': 150.0, 'USD/EUR': 0.92, 'USD/GBP': 0.80}
        self.predicted_prices = {'USD/JPY': 150.0, 'USD/EUR': 0.92, 'USD/GBP': 0.80}
        self.price_history = {pair: [self.current_prices[pair]] * 50 for pair in self.currency_pairs}
        self.timestamps = [datetime.now() - timedelta(minutes=i) for i in range(49, -1, -1)]
        
        # Trading environment variables
        self.initial_capital = 3000.0
        self.capital = {'USD/JPY': 1000.0, 'USD/EUR': 1000.0, 'USD/GBP': 1000.0}
        self.positions = {'USD/JPY': 0.0, 'USD/EUR': 0.0, 'USD/GBP': 0.0}
        self.entry_prices = {'USD/JPY': 0.0, 'USD/EUR': 0.0, 'USD/GBP': 0.0}
        self.floating_pnl = {'USD/JPY': 0.0, 'USD/EUR': 0.0, 'USD/GBP': 0.0}
        
        # Trading parameters
        self.leverage = 5
        self.margin_per_lot = 10
        self.liquidation_threshold = 0.3
        
        # Other information
        self.weather_info = "Loading..."
        self.gold_price = 2000.0
        self.silver_price = 25.0
        
    def initialize_predictors(self):
        """Initialize prediction models"""
        self.predictors = {}
        for pair in self.currency_pairs:
            self.predictors[pair] = SimpleForexPredictor()
            # Train model with historical data
            historical_data = np.array(self.price_history[pair])
            self.predictors[pair].train(historical_data)
        
    def create_widgets(self):
        """Create all GUI components"""
        self.create_header()
        self.create_main_content()
        
    def create_header(self):
        """Create header area"""
        header_frame = tk.Frame(self.root, bg='#2d2d2d', height=50)
        header_frame.pack(fill='x', padx=10, pady=5)
        header_frame.pack_propagate(False)
        
        # Main title
        title_label = tk.Label(header_frame, text="🌍 Multi-Functional Forex Trading Simulation System", 
                              font=self.title_font, fg='#00ff88', bg='#2d2d2d')
        title_label.pack(side='left', padx=20, pady=5)
        
        # Real-time clock
        self.time_label = tk.Label(header_frame, text="", font=self.normal_font, 
                                  fg='#ffffff', bg='#2d2d2d')
        self.time_label.pack(side='right', padx=20, pady=5)
        
    def create_main_content(self):
        """Create main content area"""
        main_frame = tk.Frame(self.root, bg='#1e1e1e')
        main_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Left panel - Market information
        left_panel = tk.Frame(main_frame, bg='#2d2d2d', width=400)
        left_panel.pack(side='left', fill='y', padx=5)
        left_panel.pack_propagate(False)
        
        self.create_market_info_panel(left_panel)
        
        # Right panel - Trading and charts
        right_panel = tk.Frame(main_frame, bg='#1e1e1e')
        right_panel.pack(side='right', fill='both', expand=True, padx=5)
        
        self.create_trading_panel(right_panel)
        self.create_charts_panel(right_panel)
        self.create_status_bar()
        
    def create_market_info_panel(self, parent):
        """Create market information panel"""
        # Exchange rates information
        rates_frame = tk.LabelFrame(parent, text="📈 Real-time Exchange Rates", font=self.header_font,
                                   fg='#00ff88', bg='#2d2d2d', bd=2)
        rates_frame.pack(fill='x', padx=10, pady=10)
        
        self.rate_labels = {}
        self.prediction_labels = {}
        
        for pair in self.currency_pairs:
            pair_frame = tk.Frame(rates_frame, bg='#2d2d2d')
            pair_frame.pack(fill='x', padx=5, pady=5)
            
            tk.Label(pair_frame, text=pair, font=self.header_font, 
                    fg='#ffffff', bg='#2d2d2d').pack(side='left')
            
            self.rate_labels[pair] = tk.Label(pair_frame, text="0.0000", 
                                            font=self.normal_font, fg='#00ff88', bg='#2d2d2d')
            self.rate_labels[pair].pack(side='right')
            
            # Predicted price
            pred_frame = tk.Frame(rates_frame, bg='#2d2d2d')
            pred_frame.pack(fill='x', padx=5, pady=2)
            
            tk.Label(pred_frame, text=f"AI Prediction {pair}:", font=self.normal_font, 
                    fg='#cccccc', bg='#2d2d2d').pack(side='left')
            
            self.prediction_labels[pair] = tk.Label(pred_frame, text="0.0000", 
                                                  font=self.normal_font, fg='#ffaa00', bg='#2d2d2d')
            self.prediction_labels[pair].pack(side='right')
        
        # Weather information
        weather_frame = tk.LabelFrame(parent, text="🌤️ Weather Info (Tainan)", font=self.header_font,
                                     fg='#00ff88', bg='#2d2d2d', bd=2)
        weather_frame.pack(fill='x', padx=10, pady=10)
        
        self.weather_label = tk.Label(weather_frame, text=self.weather_info, 
                                     font=self.normal_font, fg='#ffffff', bg='#2d2d2d',
                                     wraplength=350, justify='left')
        self.weather_label.pack(padx=10, pady=10)
        
        # Precious metals prices
        metals_frame = tk.LabelFrame(parent, text="🥇 Precious Metals", font=self.header_font,
                                    fg='#00ff88', bg='#2d2d2d', bd=2)
        metals_frame.pack(fill='x', padx=10, pady=10)
        
        self.gold_label = tk.Label(metals_frame, text=f"Gold: ${self.gold_price:.2f}/oz", 
                                  font=self.normal_font, fg='#ffd700', bg='#2d2d2d')
        self.gold_label.pack(padx=10, pady=5)
        
        self.silver_label = tk.Label(metals_frame, text=f"Silver: ${self.silver_price:.2f}/oz", 
                                    font=self.normal_font, fg='#c0c0c0', bg='#2d2d2d')
        self.silver_label.pack(padx=10, pady=5)
        
        # Trading parameters setup
        self.create_parameter_panel(parent)
        
    def create_parameter_panel(self, parent):
        """Create parameter setup panel"""
        params_frame = tk.LabelFrame(parent, text="⚙️ Trading Parameters", font=self.header_font,
                                    fg='#00ff88', bg='#2d2d2d', bd=2)
        params_frame.pack(fill='x', padx=10, pady=10)
        
        # Leverage setting
        tk.Label(params_frame, text="Leverage:", font=self.normal_font, 
                fg='#ffffff', bg='#2d2d2d').pack(anchor='w', padx=10, pady=2)
        
        self.leverage_var = tk.StringVar(value=str(self.leverage))
        leverage_entry = tk.Entry(params_frame, textvariable=self.leverage_var, 
                                 font=self.normal_font, width=10)
        leverage_entry.pack(padx=10, pady=2)
        
        # Margin setting
        tk.Label(params_frame, text="Margin per Lot:", font=self.normal_font, 
                fg='#ffffff', bg='#2d2d2d').pack(anchor='w', padx=10, pady=2)
        
        self.margin_var = tk.StringVar(value=str(self.margin_per_lot))
        margin_entry = tk.Entry(params_frame, textvariable=self.margin_var, 
                               font=self.normal_font, width=10)
        margin_entry.pack(padx=10, pady=2)
        
        # Update parameters button
        update_btn = tk.Button(params_frame, text="Update Parameters", command=self.update_parameters,
                              bg='#0066cc', fg='white', font=self.normal_font)
        update_btn.pack(pady=10)
        
    def create_trading_panel(self, parent):
        """Create trading panel"""
        # Trading control panel
        trading_frame = tk.LabelFrame(parent, text="💼 Trading Control", font=self.header_font,
                                     fg='#00ff88', bg='#2d2d2d', bd=2)
        trading_frame.pack(fill='x', padx=10, pady=10)
        
        # Currency pair selection
        pair_frame = tk.Frame(trading_frame, bg='#2d2d2d')
        pair_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Label(pair_frame, text="Select Currency Pair:", font=self.normal_font, 
                fg='#ffffff', bg='#2d2d2d').pack(side='left')
        
        self.selected_pair = tk.StringVar(value=self.currency_pairs[0])
        pair_combo = ttk.Combobox(pair_frame, textvariable=self.selected_pair, 
                                 values=self.currency_pairs, state='readonly')
        pair_combo.pack(side='right', padx=10)
        pair_combo.bind('<<ComboboxSelected>>', self.on_pair_selected)
        
        # Trade amount
        amount_frame = tk.Frame(trading_frame, bg='#2d2d2d')
        amount_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Label(amount_frame, text="Trade Amount (Lots):", font=self.normal_font, 
                fg='#ffffff', bg='#2d2d2d').pack(side='left')
        
        self.trade_amount = tk.StringVar(value="1")
        amount_entry = tk.Entry(amount_frame, textvariable=self.trade_amount, 
                               font=self.normal_font, width=10)
        amount_entry.pack(side='right', padx=10)
        
        # Trading buttons
        button_frame = tk.Frame(trading_frame, bg='#2d2d2d')
        button_frame.pack(fill='x', padx=10, pady=10)
        
        buy_btn = tk.Button(button_frame, text="📈 Buy (Long)", command=self.buy_position,
                           bg='#00aa00', fg='white', font=self.header_font, width=15)
        buy_btn.pack(side='left', padx=5)
        
        sell_btn = tk.Button(button_frame, text="📉 Sell (Short)", command=self.sell_position,
                            bg='#aa0000', fg='white', font=self.header_font, width=15)
        sell_btn.pack(side='left', padx=5)
        
        close_btn = tk.Button(button_frame, text="🔒 Close Position", command=self.close_position,
                             bg='#ff6600', fg='white', font=self.header_font, width=15)
        close_btn.pack(side='left', padx=5)
        
        # Account information
        self.create_account_panel(parent)
        
    def create_account_panel(self, parent):
        """Create account information panel"""
        account_frame = tk.LabelFrame(parent, text="💰 Account Information", font=self.header_font,
                                     fg='#00ff88', bg='#2d2d2d', bd=2)
        account_frame.pack(fill='x', padx=10, pady=10)
        
        self.account_labels = {}
        for pair in self.currency_pairs:
            pair_account_frame = tk.Frame(account_frame, bg='#2d2d2d')
            pair_account_frame.pack(fill='x', padx=10, pady=5)
            
            tk.Label(pair_account_frame, text=f"{pair}:", font=self.header_font, 
                    fg='#ffffff', bg='#2d2d2d').pack(side='left')
            
            self.account_labels[pair] = tk.Label(pair_account_frame, 
                                               text="Capital: $1000 | Position: 0 | Float P&L: $0", 
                                               font=self.normal_font, fg='#ffffff', bg='#2d2d2d')
            self.account_labels[pair].pack(side='right')
        
        # Total assets
        total_frame = tk.Frame(account_frame, bg='#2d2d2d')
        total_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Label(total_frame, text="Total Assets:", font=self.header_font, 
                fg='#ffffff', bg='#2d2d2d').pack(side='left')
        
        self.total_asset_label = tk.Label(total_frame, text="$3000.00", 
                                         font=self.header_font, fg='#00ff88', bg='#2d2d2d')
        self.total_asset_label.pack(side='right')
        
    def create_charts_panel(self, parent):
        """Create charts panel"""
        charts_frame = tk.LabelFrame(parent, text="📊 Historical Exchange Rate Charts", font=self.header_font,
                                    fg='#00ff88', bg='#2d2d2d', bd=2)
        charts_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(12, 8), facecolor='#2d2d2d')
        self.fig.suptitle('Real-time Exchange Rate Trends', color='white', fontsize=14)
        
        # Create subplots for each currency pair
        self.axes = {}
        for i, pair in enumerate(self.currency_pairs):
            ax = self.fig.add_subplot(3, 1, i+1)
            ax.set_facecolor('#1e1e1e')
            ax.tick_params(colors='white')
            ax.set_ylabel(f'{pair} Rate', color='white')
            ax.grid(True, alpha=0.3)
            self.axes[pair] = ax
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, charts_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=5, pady=5)
        
        # Chart control buttons
        chart_control_frame = tk.Frame(charts_frame, bg='#2d2d2d')
        chart_control_frame.pack(fill='x', padx=5, pady=5)
        
        refresh_btn = tk.Button(chart_control_frame, text="🔄 Refresh Charts", 
                               command=self.update_charts, bg='#0066cc', fg='white')
        refresh_btn.pack(side='left', padx=5)
        
        auto_scale_btn = tk.Button(chart_control_frame, text="📏 Auto Scale", 
                                  command=self.auto_scale_charts, bg='#0066cc', fg='white')
        auto_scale_btn.pack(side='left', padx=5)
        
    def create_status_bar(self):
        """Create status bar"""
        status_frame = tk.Frame(self.root, bg='#2d2d2d', height=30)
        status_frame.pack(fill='x', side='bottom')
        status_frame.pack_propagate(False)
        
        self.status_label = tk.Label(status_frame, text="System Ready - Using Simplified AI Prediction Model", 
                                    font=self.normal_font, fg='#ffffff', bg='#2d2d2d')
        self.status_label.pack(side='left', padx=10, pady=5)
        
        self.ai_status_label = tk.Label(status_frame, text="AI Prediction: Running", 
                                       font=self.normal_font, fg='#ffaa00', bg='#2d2d2d')
        self.ai_status_label.pack(side='right', padx=10, pady=5)
        
    def setup_trading_environment(self):
        """Setup trading environment"""
        self.fx_trading = FXTradingEnvironment(
            initial_capital=self.initial_capital,
            leverage=self.leverage,
            margin_per_lot=self.margin_per_lot,
            liquidation_threshold=self.liquidation_threshold
        )
        
    def start_real_time_updates(self):
        """Start real-time updates"""
        self.update_time()
        self.update_market_data()
        self.update_ai_predictions()
        self.update_account_info()
        self.update_charts()
        
        # Update every 2 seconds
        self.root.after(2000, self.start_real_time_updates)
        
    def update_time(self):
        """Update time display"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.time_label.config(text=f"🕒 {current_time}")
        
    def update_market_data(self):
        """Update market data"""
        try:
            current_time = datetime.now()
            
            for pair in self.currency_pairs:
                # Simulate price fluctuation
                current = self.current_prices[pair]
                volatility = 0.001 if 'JPY' in pair else 0.0001
                change = np.random.normal(0, volatility) * current
                new_price = max(0.1, current + change)
                
                self.current_prices[pair] = new_price
                self.price_history[pair].append(new_price)
                
                # Maintain history length
                if len(self.price_history[pair]) > 100:
                    self.price_history[pair] = self.price_history[pair][-100:]
                
                # Update display
                self.rate_labels[pair].config(text=f"{new_price:.4f}")
            
            # Update timestamps
            self.timestamps.append(current_time)
            if len(self.timestamps) > 100:
                self.timestamps = self.timestamps[-100:]
                
            # Update weather and metals prices
            self.update_weather_info()
            self.update_metals_prices()
            
        except Exception as e:
            self.status_label.config(text=f"Market data update error: {str(e)}")
            
    def update_weather_info(self):
        """Update weather information"""
        weather_conditions = ["Sunny☀️", "Cloudy⛅", "Light Rain🌧️", "Overcast☁️"]
        temperature = np.random.randint(22, 32)
        condition = np.random.choice(weather_conditions)
        humidity = np.random.randint(60, 85)
        
        weather_text = f"Tainan City: {condition}\nTemperature: {temperature}°C\nHumidity: {humidity}%"
        self.weather_label.config(text=weather_text)
        
    def update_metals_prices(self):
        """Update precious metals prices"""
        self.gold_price += np.random.normal(0, 3)
        self.silver_price += np.random.normal(0, 0.3)
        
        self.gold_price = max(1800, self.gold_price)
        self.silver_price = max(20, self.silver_price)
        
        self.gold_label.config(text=f"Gold: ${self.gold_price:.2f}/oz")
        self.silver_label.config(text=f"Silver: ${self.silver_price:.2f}/oz")
        
    def update_ai_predictions(self):
        """Update AI predictions"""
        try:
            for pair in self.currency_pairs:
                # Use simplified prediction model
                historical_data = np.array(self.price_history[pair])
                predicted_price = self.predictors[pair].predict(historical_data)
                
                self.predicted_prices[pair] = predicted_price
                self.prediction_labels[pair].config(text=f"{predicted_price:.4f}")
                
                # Retrain model (online learning)
                if len(self.price_history[pair]) > 20:
                    self.predictors[pair].train(historical_data[-50:])
                
            self.ai_status_label.config(text="AI Prediction: Running ✓")
            
        except Exception as e:
            self.ai_status_label.config(text=f"AI Prediction Error: {str(e)}")
            
    def update_charts(self):
        """Update historical charts"""
        try:
            for pair in self.currency_pairs:
                ax = self.axes[pair]
                ax.clear()
                
                # Plot historical data
                if len(self.price_history[pair]) > 1 and len(self.timestamps) > 1:
                    # Ensure data lengths match
                    min_len = min(len(self.price_history[pair]), len(self.timestamps))
                    prices = self.price_history[pair][-min_len:]
                    times = self.timestamps[-min_len:]
                    
                    # Plot price line
                    ax.plot(times, prices, color='#00ff88', linewidth=2, label='Actual Price')
                    
                    # Plot prediction if available
                    if pair in self.predicted_prices:
                        pred_price = self.predicted_prices[pair]
                        ax.axhline(y=pred_price, color='#ffaa00', linestyle='--', 
                                  alpha=0.7, label='AI Prediction')
                
                # Styling
                ax.set_facecolor('#1e1e1e')
                ax.tick_params(colors='white', labelsize=8)
                ax.set_ylabel(f'{pair} Rate', color='white', fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.legend(loc='upper left', fontsize=8)
                
                # Format x-axis for time
                if len(self.timestamps) > 1:
                    ax.tick_params(axis='x', rotation=45)
            
            self.fig.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            print(f"Chart update error: {e}")
            
    def auto_scale_charts(self):
        """Auto scale all charts"""
        for pair in self.currency_pairs:
            ax = self.axes[pair]
            ax.relim()
            ax.autoscale_view()
        self.canvas.draw()
        
    def on_pair_selected(self, event=None):
        """Handle currency pair selection"""
        selected = self.selected_pair.get()
        self.status_label.config(text=f"Selected currency pair: {selected}")
        
    def update_account_info(self):
        """Update account information"""
        total_asset = 0
        
        for pair in self.currency_pairs:
            if self.positions[pair] != 0:
                current_price = self.current_prices[pair]
                entry_price = self.entry_prices[pair]
                
                if entry_price > 0:
                    price_change_pct = (current_price - entry_price) / entry_price
                    self.floating_pnl[pair] = self.positions[pair] * price_change_pct * self.margin_per_lot * self.leverage
                    
            account_value = self.capital[pair] + self.floating_pnl[pair]
            total_asset += account_value
            
            self.account_labels[pair].config(
                text=f"Capital: ${self.capital[pair]:.2f} | Position: {self.positions[pair]:.1f} | Float P&L: ${self.floating_pnl[pair]:.2f}"
            )
            
        self.total_asset_label.config(text=f"${total_asset:.2f}")
        self.check_liquidation()
        
    def check_liquidation(self):
        """Check liquidation conditions"""
        for pair in self.currency_pairs:
            if self.positions[pair] != 0:
                account_value = self.capital[pair] + self.floating_pnl[pair]
                margin_used = abs(self.positions[pair]) * self.margin_per_lot
                
                if margin_used > 0:
                    margin_ratio = account_value / margin_used
                    
                    if margin_ratio < self.liquidation_threshold:
                        self.force_liquidation(pair)
                        
    def force_liquidation(self, pair):
        """Force liquidation"""
        if self.positions[pair] != 0:
            self.capital[pair] += self.floating_pnl[pair]
            self.positions[pair] = 0
            self.entry_prices[pair] = 0
            self.floating_pnl[pair] = 0
            
            messagebox.showwarning("Force Liquidation", f"{pair} triggered liquidation condition, position automatically closed!")
            self.status_label.config(text=f"{pair} Force liquidation executed")
            
    def buy_position(self):
        """Buy position"""
        pair = self.selected_pair.get()
        try:
            amount = float(self.trade_amount.get())
            current_price = self.current_prices[pair]
            required_margin = amount * self.margin_per_lot
            
            if self.capital[pair] >= required_margin:
                if self.positions[pair] == 0:
                    self.entry_prices[pair] = current_price
                else:
                    total_value = self.positions[pair] * self.entry_prices[pair] + amount * current_price
                    total_position = self.positions[pair] + amount
                    self.entry_prices[pair] = total_value / total_position if total_position != 0 else current_price
                    
                self.positions[pair] += amount
                self.capital[pair] -= required_margin
                
                self.status_label.config(text=f"✅ Successfully bought {amount} lots of {pair}")
            else:
                messagebox.showerror("Insufficient Funds", "Insufficient funds to open this position")
                
        except ValueError:
            messagebox.showerror("Input Error", "Please enter a valid trade amount")
            
    def sell_position(self):
        """Sell position"""
        pair = self.selected_pair.get()
        try:
            amount = float(self.trade_amount.get())
            current_price = self.current_prices[pair]
            required_margin = amount * self.margin_per_lot
            
            if self.capital[pair] >= required_margin:
                if self.positions[pair] == 0:
                    self.entry_prices[pair] = current_price
                else:
                    total_value = self.positions[pair] * self.entry_prices[pair] - amount * current_price
                    total_position = self.positions[pair] - amount
                    self.entry_prices[pair] = total_value / total_position if total_position != 0 else current_price
                    
                self.positions[pair] -= amount
                self.capital[pair] -= required_margin
                
                self.status_label.config(text=f"✅ Successfully sold {amount} lots of {pair}")
            else:
                messagebox.showerror("Insufficient Funds", "Insufficient funds to open this position")
                
        except ValueError:
            messagebox.showerror("Input Error", "Please enter a valid trade amount")
            
    def close_position(self):
        """Close position"""
        pair = self.selected_pair.get()
        
        if self.positions[pair] != 0:
            self.capital[pair] += self.floating_pnl[pair]
            realized_pnl = self.floating_pnl[pair]
            
            self.positions[pair] = 0
            self.entry_prices[pair] = 0
            self.floating_pnl[pair] = 0
            
            self.status_label.config(text=f"✅ {pair} position closed, realized P&L: ${realized_pnl:.2f}")
        else:
            messagebox.showinfo("No Position", f"No open position for {pair}")
            
    def update_parameters(self):
        """Update trading parameters"""
        try:
            new_leverage = int(self.leverage_var.get())
            new_margin = float(self.margin_var.get())
            
            if new_leverage > 0 and new_margin > 0:
                self.leverage = new_leverage
                self.margin_per_lot = new_margin
                
                self.status_label.config(text="✅ Trading parameters updated")
                messagebox.showinfo("Parameters Updated", f"Leverage: {self.leverage}x, Margin: ${self.margin_per_lot}")
            else:
                messagebox.showerror("Parameter Error", "Parameters must be greater than 0")
                
        except ValueError:
            messagebox.showerror("Parameter Error", "Please enter valid numeric values")

class FXTradingEnvironment:
    """Trading environment class"""
    def __init__(self, initial_capital, leverage, margin_per_lot, liquidation_threshold):
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.margin_per_lot = margin_per_lot
        self.liquidation_threshold = liquidation_threshold
        self.reset()
        
    def reset(self):
        """Reset trading environment"""
        self.capital = {'USD/JPY': 1000.0, 'USD/EUR': 1000.0, 'USD/GBP': 1000.0}
        self.positions = {'USD/JPY': 0.0, 'USD/EUR': 0.0, 'USD/GBP': 0.0}
        self.entry_prices = {'USD/JPY': 0.0, 'USD/EUR': 0.0, 'USD/GBP': 0.0}

def main():
    """Main function"""
    root = tk.Tk()
    app = ForexTradingGUI(root)
    
    def on_closing():
        if messagebox.askokcancel("Exit", "Are you sure you want to exit the trading system?"):
            root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("Program interrupted by user")
    except Exception as e:
        print(f"Program execution error: {e}")

if __name__ == "__main__":
    main()
