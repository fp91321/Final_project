import tkinter as tk
from tkinter import ttk, messagebox
import tkinter.font as tkFont
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime
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
class FreeWeatherProvider:
    """免費天氣資料提供者"""
    
    def __init__(self):
        self.cache = {}
        self.last_update = 0
        self.cache_duration = 600  # 10分鐘緩存
        
    def get_weather_data(self, city="Tainan"):
        """獲取天氣資料 - 使用多個免費API"""
        current_time = time.time()
        
        # 檢查緩存
        if (city in self.cache and 
            current_time - self.last_update < self.cache_duration):
            return self.cache[city]
        
        # 嘗試多個免費API
        weather_data = self._try_wttr_api(city) or self._try_7timer_api(city)
        
        if weather_data:
            self.cache[city] = weather_data
            self.last_update = current_time
            return weather_data
        
        return self._get_fallback_weather()
    
    def _try_wttr_api(self, city):
        """嘗試 wttr.in API"""
        try:
            url = f"https://wttr.in/{city}?format=j1"
            response = requests.get(url, timeout=8)
            
            if response.status_code == 200:
                data = response.json()
                current = data['current_condition'][0]
                
                return {
                    'temperature': int(current['temp_C']),
                    'condition': current['weatherDesc'][0]['value'],
                    'humidity': int(current['humidity']),
                    'feels_like': int(current['FeelsLikeC']),
                    'wind_speed': current['windspeedKmph'],
                    'source': 'wttr.in'
                }
        except Exception as e:
            print(f"wttr.in API 錯誤: {e}")
            return None
    
    def _try_7timer_api(self, city):
        """嘗試 7Timer! API (需要座標)"""
        try:
            # 台南的座標
            coords = {
                'Tainan': (22.99, 120.21),
                'Taipei': (25.04, 121.51),
                'Kaohsiung': (22.63, 120.30)
            }
            
            if city not in coords:
                return None
                
            lat, lon = coords[city]
            url = f"http://www.7timer.info/bin/api.pl?lon={lon}&lat={lat}&product=civil&output=json"
            response = requests.get(url, timeout=8)
            
            if response.status_code == 200:
                data = response.json()
                if data['dataseries']:
                    current = data['dataseries'][0]
                    
                    # 7Timer 使用不同的數據格式
                    weather_map = {
                        'clear': 'Clear',
                        'pcloudy': 'Partly Cloudy',
                        'cloudy': 'Cloudy',
                        'rain': 'Rain'
                    }
                    
                    return {
                        'temperature': current['temp2m'],
                        'condition': weather_map.get(current['weather'], 'Unknown'),
                        'humidity': 70,  # 7Timer 不提供濕度，使用預設值
                        'feels_like': current['temp2m'],
                        'wind_speed': current['wind10m']['speed'],
                        'source': '7Timer!'
                    }
        except Exception as e:
            print(f"7Timer API 錯誤: {e}")
            return None
    
    def _get_fallback_weather(self):
        """備用天氣資料"""
        return {
            'temperature': 28,
            'condition': 'Partly Cloudy',
            'humidity': 75,
            'feels_like': 30,
            'wind_speed': '10',
            'source': 'Fallback'
        }
    
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

class AdvancedRiskManager:
    """Advanced risk management system with dynamic leverage and margin calculations"""
    
    def __init__(self, initial_leverage=5, initial_margin=10, liquidation_threshold=0.3):
        self.leverage = initial_leverage
        self.margin_per_lot = initial_margin
        self.liquidation_threshold = liquidation_threshold
        self.risk_levels = {
            'low': {'max_leverage': 20, 'margin_multiplier': 1.0},
            'medium': {'max_leverage': 10, 'margin_multiplier': 1.5},
            'high': {'max_leverage': 5, 'margin_multiplier': 2.0}
        }
        
    def calculate_required_margin(self, position_size, current_price, leverage=None):
        """Calculate required margin for a position"""
        if leverage is None:
            leverage = self.leverage
        
        notional_value = abs(position_size) * current_price
        required_margin = notional_value / leverage
        return required_margin
    
    def calculate_floating_pnl(self, position_size, entry_price, current_price, leverage=None):
        """Calculate floating P&L for a position"""
        if position_size == 0 or entry_price == 0:
            return 0.0
            
        if leverage is None:
            leverage = self.leverage
            
        # Calculate price change percentage
        price_change_pct = (current_price - entry_price) / entry_price
        
        # Calculate P&L based on position size and leverage
        notional_value = abs(position_size) * entry_price
        pnl = position_size * price_change_pct * notional_value / leverage * leverage
        
        return pnl
    
    def calculate_margin_level(self, equity, used_margin):
        """Calculate margin level (equity / used_margin * 100)"""
        if used_margin == 0:
            return float('inf')
        return (equity / used_margin) * 100
    
    def check_margin_call(self, margin_level, margin_call_level=150):
        """Check if margin call should be triggered"""
        return margin_level <= margin_call_level
    
    def check_liquidation(self, margin_level):
        """Check if liquidation should be triggered"""
        liquidation_level = self.liquidation_threshold * 100
        return margin_level <= liquidation_level
    
    def get_dynamic_leverage(self, account_equity, risk_profile='medium'):
        """Get dynamic leverage based on account equity and risk profile"""
        risk_config = self.risk_levels.get(risk_profile, self.risk_levels['medium'])
        
        # Adjust leverage based on equity size
        if account_equity < 1000:
            return min(self.leverage, risk_config['max_leverage'])
        elif account_equity < 5000:
            return min(self.leverage * 0.8, risk_config['max_leverage'])
        elif account_equity < 10000:
            return min(self.leverage * 0.6, risk_config['max_leverage'])
        else:
            return min(self.leverage * 0.4, risk_config['max_leverage'])

class ForexTradingGUI:
    def __init__(self, root):
        self.root = root
        self.setup_main_window()
        self.initialize_variables()
        self.initialize_predictors()
        self.initialize_risk_manager()
        self.create_widgets()
        self.setup_trading_environment()
        self.start_real_time_updates()
        
    def setup_main_window(self):
        """Setup main window"""
        self.root.title("🌍 2025 ML Final FX-Change® - Advanced Trading System")
        self.root.geometry("1800x1200")
        self.root.configure(bg='#1e1e1e')
        
        # Setup fonts
        self.title_font = tkFont.Font(family="Arial", size=16, weight="bold")
        self.header_font = tkFont.Font(family="Arial", size=12, weight="bold")
        self.normal_font = tkFont.Font(family="Arial", size=10)
        
    def initialize_variables(self):
        """Initialize variables"""
        self.currency_pairs = ['USD/JPY', 'USD/EUR', 'USD/GBP']
        self.current_prices = {'USD/JPY': 150.0, 'USD/EUR': 0.92, 'USD/GBP': 0.80}
        self.predicted_prices = {'USD/JPY': 150.0, 'USD/EUR': 0.92, 'USD/GBP': 0.80}
        self.price_history = {pair: [self.current_prices[pair]] * 50 for pair in self.currency_pairs}
        self.timestamps = [datetime.now() - timedelta(minutes=i) for i in range(49, -1, -1)]
        
        # Enhanced trading environment variables
        self.initial_capital = 3000.0
        self.capital = {'USD/JPY': 1000.0, 'USD/EUR': 1000.0, 'USD/GBP': 1000.0}
        self.positions = {'USD/JPY': 0.0, 'USD/EUR': 0.0, 'USD/GBP': 0.0}
        self.entry_prices = {'USD/JPY': 0.0, 'USD/EUR': 0.0, 'USD/GBP': 0.0}
        self.floating_pnl = {'USD/JPY': 0.0, 'USD/EUR': 0.0, 'USD/GBP': 0.0}
        self.used_margin = {'USD/JPY': 0.0, 'USD/EUR': 0.0, 'USD/GBP': 0.0}
        self.margin_level = {'USD/JPY': float('inf'), 'USD/EUR': float('inf'), 'USD/GBP': float('inf')}
        
        # Advanced trading parameters
        self.leverage = 5
        self.margin_per_lot = 10
        self.liquidation_threshold = 0.3
        self.margin_call_level = 150  # 150%
        self.dynamic_leverage_enabled = True
        self.risk_profile = 'medium'
        
        # Trading history
        self.trade_history = []
        self.liquidation_history = []
        
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
    
    def initialize_risk_manager(self):
        """Initialize advanced risk management system"""
        self.risk_manager = AdvancedRiskManager(
            initial_leverage=self.leverage,
            initial_margin=self.margin_per_lot,
            liquidation_threshold=self.liquidation_threshold
        )
        
    def create_widgets(self):
        """Create all GUI components"""
        self.create_header()
        self.create_main_content()
        
    def create_header(self):
        """Create header area"""
        header_frame = tk.Frame(self.root, bg='#2d2d2d', height=60)
        header_frame.pack(fill='x', padx=10, pady=5)
        header_frame.pack_propagate(False)
        
        # Main title
        title_label = tk.Label(header_frame, text="🌍 Advanced Forex Trading System with Dynamic Risk Management", 
                              font=self.title_font, fg='#00ff88', bg='#2d2d2d')
        title_label.pack(side='left', padx=20, pady=10)
        
        # Real-time clock
        self.time_label = tk.Label(header_frame, text="", font=self.normal_font, 
                                  fg='#ffffff', bg='#2d2d2d')
        self.time_label.pack(side='right', padx=20, pady=10)
        
    def create_main_content(self):
        """Create main content area"""
        main_frame = tk.Frame(self.root, bg='#1e1e1e')
        main_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Left panel - Market information and risk management
        left_panel = tk.Frame(main_frame, bg='#2d2d2d', width=450)
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
        
        # Advanced Risk Management Panel
        self.create_risk_management_panel(parent)
        
        # Weather information
        weather_frame = tk.LabelFrame(parent, text="🌤️ Weather Info (Tainan)", font=self.header_font,
                                     fg='#00ff88', bg='#2d2d2d', bd=2)
        weather_frame.pack(fill='x', padx=10, pady=10)
        
        self.weather_label = tk.Label(weather_frame, text=self.weather_info, 
                                     font=self.normal_font, fg='#ffffff', bg='#2d2d2d',
                                     wraplength=350, justify='left')
        self.weather_label.pack(padx=10, pady=10)
        """
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
        """
        # Trading parameters setup
        self.create_parameter_panel(parent)
    
    def create_risk_management_panel(self, parent):
        """Create advanced risk management panel"""
        risk_frame = tk.LabelFrame(parent, text="⚠️ Risk Management Dashboard", font=self.header_font,
                                  fg='#ff6600', bg='#2d2d2d', bd=2)
        risk_frame.pack(fill='x', padx=10, pady=10)
        
        # Margin levels for each currency pair
        self.margin_level_labels = {}
        self.used_margin_labels = {}
        
        for pair in self.currency_pairs:
            pair_risk_frame = tk.Frame(risk_frame, bg='#2d2d2d')
            pair_risk_frame.pack(fill='x', padx=5, pady=3)
            
            tk.Label(pair_risk_frame, text=f"{pair} Margin Level:", font=self.normal_font, 
                    fg='#ffffff', bg='#2d2d2d').pack(side='left')
            
            self.margin_level_labels[pair] = tk.Label(pair_risk_frame, text="∞%", 
                                                    font=self.normal_font, fg='#00ff88', bg='#2d2d2d')
            self.margin_level_labels[pair].pack(side='right')
            
            # Used margin display
            margin_frame = tk.Frame(risk_frame, bg='#2d2d2d')
            margin_frame.pack(fill='x', padx=5, pady=2)
            
            tk.Label(margin_frame, text=f"{pair} Used Margin:", font=self.normal_font, 
                    fg='#cccccc', bg='#2d2d2d').pack(side='left')
            
            self.used_margin_labels[pair] = tk.Label(margin_frame, text="$0.00", 
                                                   font=self.normal_font, fg='#ffffff', bg='#2d2d2d')
            self.used_margin_labels[pair].pack(side='right')
        
        # Overall risk indicators
        overall_risk_frame = tk.Frame(risk_frame, bg='#2d2d2d')
        overall_risk_frame.pack(fill='x', padx=5, pady=5)
        
        tk.Label(overall_risk_frame, text="Overall Risk Level:", font=self.header_font, 
                fg='#ffffff', bg='#2d2d2d').pack(side='left')
        
        self.overall_risk_label = tk.Label(overall_risk_frame, text="LOW", 
                                         font=self.header_font, fg='#00ff88', bg='#2d2d2d')
        self.overall_risk_label.pack(side='right')
        
    def create_parameter_panel(self, parent):
        """Create enhanced parameter setup panel"""
        params_frame = tk.LabelFrame(parent, text="⚙️ Advanced Trading Parameters", font=self.header_font,
                                    fg='#00ff88', bg='#2d2d2d', bd=2)
        params_frame.pack(fill='x', padx=10, pady=10)
        
        # Leverage setting
        tk.Label(params_frame, text="Base Leverage:", font=self.normal_font, 
                fg='#ffffff', bg='#2d2d2d').pack(anchor='w', padx=10, pady=2)
        
        self.leverage_var = tk.StringVar(value=str(self.leverage))
        leverage_entry = tk.Entry(params_frame, textvariable=self.leverage_var, 
                                 font=self.normal_font, width=10)
        leverage_entry.pack(padx=10, pady=2)
        
        # Margin setting
        tk.Label(params_frame, text="Margin per Lot ($):", font=self.normal_font, 
                fg='#ffffff', bg='#2d2d2d').pack(anchor='w', padx=10, pady=2)
        
        self.margin_var = tk.StringVar(value=str(self.margin_per_lot))
        margin_entry = tk.Entry(params_frame, textvariable=self.margin_var, 
                               font=self.normal_font, width=10)
        margin_entry.pack(padx=10, pady=2)
        
        # Liquidation threshold setting
        tk.Label(params_frame, text="Liquidation Threshold (%):", font=self.normal_font, 
                fg='#ffffff', bg='#2d2d2d').pack(anchor='w', padx=10, pady=2)
        
        self.liquidation_var = tk.StringVar(value=str(int(self.liquidation_threshold * 100)))
        liquidation_entry = tk.Entry(params_frame, textvariable=self.liquidation_var, 
                                   font=self.normal_font, width=10)
        liquidation_entry.pack(padx=10, pady=2)
        
        # Dynamic leverage toggle
        self.dynamic_leverage_var = tk.BooleanVar(value=self.dynamic_leverage_enabled)
        dynamic_check = tk.Checkbutton(params_frame, text="Enable Dynamic Leverage", 
                                     variable=self.dynamic_leverage_var,
                                     font=self.normal_font, fg='#ffffff', bg='#2d2d2d',
                                     selectcolor='#1e1e1e')
        dynamic_check.pack(padx=10, pady=5)
        
        # Risk profile selection
        tk.Label(params_frame, text="Risk Profile:", font=self.normal_font, 
                fg='#ffffff', bg='#2d2d2d').pack(anchor='w', padx=10, pady=2)
        
        self.risk_profile_var = tk.StringVar(value=self.risk_profile)
        risk_combo = ttk.Combobox(params_frame, textvariable=self.risk_profile_var, 
                                 values=['low', 'medium', 'high'], state='readonly', width=10)
        risk_combo.pack(padx=10, pady=2)
        
        # Update parameters button
        update_btn = tk.Button(params_frame, text="Update Parameters", command=self.update_parameters,
                              bg='#0066cc', fg='white', font=self.normal_font)
        update_btn.pack(pady=10)
        
    def create_trading_panel(self, parent):
        """Create enhanced trading panel"""
        # Trading control panel
        trading_frame = tk.LabelFrame(parent, text="💼 Advanced Trading Control", font=self.header_font,
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
        
        # Current leverage display
        leverage_display_frame = tk.Frame(trading_frame, bg='#2d2d2d')
        leverage_display_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Label(leverage_display_frame, text="Current Leverage:", font=self.normal_font, 
                fg='#ffffff', bg='#2d2d2d').pack(side='left')
        
        self.current_leverage_label = tk.Label(leverage_display_frame, text="5x", 
                                             font=self.normal_font, fg='#ffaa00', bg='#2d2d2d')
        self.current_leverage_label.pack(side='right')
        
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
        """Create enhanced account information panel"""
        account_frame = tk.LabelFrame(parent, text="💰 Advanced Account Information", font=self.header_font,
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
        
        # Total assets and equity
        total_frame = tk.Frame(account_frame, bg='#2d2d2d')
        total_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Label(total_frame, text="Total Equity:", font=self.header_font, 
                fg='#ffffff', bg='#2d2d2d').pack(side='left')
        
        self.total_asset_label = tk.Label(total_frame, text="$3000.00", 
                                         font=self.header_font, fg='#00ff88', bg='#2d2d2d')
        self.total_asset_label.pack(side='right')
        
        # Free margin display
        free_margin_frame = tk.Frame(account_frame, bg='#2d2d2d')
        free_margin_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Label(free_margin_frame, text="Free Margin:", font=self.header_font, 
                fg='#ffffff', bg='#2d2d2d').pack(side='left')
        
        self.free_margin_label = tk.Label(free_margin_frame, text="$3000.00", 
                                         font=self.header_font, fg='#00aaff', bg='#2d2d2d')
        self.free_margin_label.pack(side='right')
        
    def create_charts_panel(self, parent):
        """Create charts panel"""
        charts_frame = tk.LabelFrame(parent, text="📊 Historical Exchange Rate Charts", font=self.header_font,
                                    fg='#00ff88', bg='#2d2d2d', bd=2)
        charts_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(14, 10), facecolor='#2d2d2d')
        self.fig.suptitle('Real-time Exchange Rate Trends with Risk Indicators', color='white', fontsize=14)
        
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
        """Create enhanced status bar"""
        status_frame = tk.Frame(self.root, bg='#2d2d2d', height=40)
        status_frame.pack(fill='x', side='bottom')
        status_frame.pack_propagate(False)
        
        self.status_label = tk.Label(status_frame, text="System Ready - Advanced Risk Management Active", 
                                    font=self.normal_font, fg='#ffffff', bg='#2d2d2d')
        self.status_label.pack(side='left', padx=10, pady=10)
        
        self.ai_status_label = tk.Label(status_frame, text="AI Prediction: Running", 
                                       font=self.normal_font, fg='#ffaa00', bg='#2d2d2d')
        self.ai_status_label.pack(side='right', padx=10, pady=10)
        
    def setup_trading_environment(self):
        """Setup enhanced trading environment"""
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
        self.update_risk_management()
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
                volatility = 0.002 if 'JPY' in pair else 0.0002  # Increased volatility for testing
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
            #self.update_metals_prices()
            
        except Exception as e:
            self.status_label.config(text=f"Market data update error: {str(e)}")
            
    def update_weather_info(self):
        """Free API"""
        try:
            if not hasattr(self, 'free_weather_provider'):
                self.free_weather_provider = FreeWeatherProvider()
            
            weather_data = self.free_weather_provider.get_weather_data("Tainan")
            
            # 天氣狀況對應表情符號
            condition_emojis = {
                'Clear': '☀️',
                'Sunny': '☀️', 
                'Partly Cloudy': '⛅',
                'Partly cloudy': '⛅',
                'Cloudy': '☁️',
                'Overcast': '☁️',
                'Rain': '🌧️',
                'Light rain': '🌧️',
                'Heavy rain': '⛈️',
                'Snow': '❄️',
                'Fog': '🌫️'
            }
            
            condition = weather_data['condition']
            emoji = condition_emojis.get(condition, '🌤️')
            
            weather_text = (f"Tainan: {condition} {emoji}\tTemperature: {weather_data['temperature']}°C\n"
                        f"Feels like: {weather_data['feels_like']}°C\t\tHumidity: {weather_data['humidity']}%\n"
                        f"Source: {weather_data['source']}")
            
            self.weather_label.config(text=weather_text)
            
        except Exception as e:
            print(f"天氣更新錯誤: {e}")
            # 使用備用方法
            self.update_weather_info_fallback()

    def update_weather_info_fallback(self):
        """備用天氣更新方法"""
        import numpy as np
        weather_conditions = ["Sunny☀️", "Cloudy⛅", "Light Rain🌧️", "Overcast☁️"]
        temperature = np.random.randint(22, 32)
        condition = np.random.choice(weather_conditions)
        humidity = np.random.randint(60, 85)
        
        weather_text = f"Tainan: {condition}\nTemperature: {temperature}°C\nHumidity: {humidity}%\n(Simulated)"
        self.weather_label.config(text=weather_text)

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
    
    def update_risk_management(self):
        """Update risk management calculations"""
        try:
            total_equity = 0
            total_used_margin = 0
            overall_risk_level = "LOW"
            
            for pair in self.currency_pairs:
                # Calculate floating P&L using advanced risk manager
                current_price = self.current_prices[pair]
                entry_price = self.entry_prices[pair]
                position_size = self.positions[pair]
                
                # Get dynamic leverage if enabled
                account_equity = self.capital[pair] + self.floating_pnl[pair]
                if self.dynamic_leverage_enabled:
                    effective_leverage = self.risk_manager.get_dynamic_leverage(
                        account_equity, self.risk_profile
                    )
                else:
                    effective_leverage = self.leverage
                
                # Calculate floating P&L
                if position_size != 0 and entry_price > 0:
                    self.floating_pnl[pair] = self.risk_manager.calculate_floating_pnl(
                        position_size, entry_price, current_price, effective_leverage
                    )
                    
                    # Calculate used margin
                    self.used_margin[pair] = self.risk_manager.calculate_required_margin(
                        position_size, current_price, effective_leverage
                    )
                else:
                    self.floating_pnl[pair] = 0.0
                    self.used_margin[pair] = 0.0
                
                # Calculate margin level
                equity = self.capital[pair] + self.floating_pnl[pair]
                if self.used_margin[pair] > 0:
                    self.margin_level[pair] = self.risk_manager.calculate_margin_level(
                        equity, self.used_margin[pair]
                    )
                else:
                    self.margin_level[pair] = float('inf')
                
                # Update displays
                if self.margin_level[pair] == float('inf'):
                    margin_text = "∞%"
                    margin_color = '#00ff88'
                else:
                    margin_text = f"{self.margin_level[pair]:.1f}%"
                    if self.margin_level[pair] <= 30:
                        margin_color = '#ff0000'  # Red for danger
                        overall_risk_level = "HIGH"
                    elif self.margin_level[pair] <= 150:
                        margin_color = '#ffaa00'  # Orange for warning
                        if overall_risk_level != "HIGH":
                            overall_risk_level = "MEDIUM"
                    else:
                        margin_color = '#00ff88'  # Green for safe
                
                self.margin_level_labels[pair].config(text=margin_text, fg=margin_color)
                self.used_margin_labels[pair].config(text=f"${self.used_margin[pair]:.2f}")
                
                total_equity += equity
                total_used_margin += self.used_margin[pair]
            
            # Update overall risk level
            risk_colors = {'LOW': '#00ff88', 'MEDIUM': '#ffaa00', 'HIGH': '#ff0000'}
            self.overall_risk_label.config(text=overall_risk_level, 
                                         fg=risk_colors.get(overall_risk_level, '#ffffff'))
            
            # Update current leverage display
            selected_pair = self.selected_pair.get()
            if self.dynamic_leverage_enabled:
                account_equity = self.capital[selected_pair] + self.floating_pnl[selected_pair]
                current_leverage = self.risk_manager.get_dynamic_leverage(
                    account_equity, self.risk_profile
                )
            else:
                current_leverage = self.leverage
            
            self.current_leverage_label.config(text=f"{current_leverage:.1f}x")
            
            # Calculate free margin
            free_margin = total_equity - total_used_margin
            self.free_margin_label.config(text=f"${free_margin:.2f}")
            
        except Exception as e:
            print(f"Risk management update error: {e}")
            
    def update_charts(self):
        """Update historical charts with risk indicators"""
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
                    
                    # Add risk indicators
                    if self.positions[pair] != 0:
                        entry_price = self.entry_prices[pair]
                        if entry_price > 0:
                            ax.axhline(y=entry_price, color='#ffffff', linestyle=':', 
                                      alpha=0.8, label='Entry Price')
                            
                            # Add liquidation level indicator
                            margin_level = self.margin_level[pair]
                            if margin_level != float('inf') and margin_level <= 50:
                                ax.fill_between(times, min(prices), max(prices), 
                                               alpha=0.1, color='red', label='Risk Zone')
                
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
        """Update enhanced account information"""
        total_equity = 0
        
        for pair in self.currency_pairs:
            equity = self.capital[pair] + self.floating_pnl[pair]
            total_equity += equity
            
            self.account_labels[pair].config(
                text=f"Capital: ${self.capital[pair]:.2f} | Position: {self.positions[pair]:.1f} | Float P&L: ${self.floating_pnl[pair]:.2f}"
            )
            
        self.total_asset_label.config(text=f"${total_equity:.2f}")
        self.check_liquidation()
        
    def check_liquidation(self):
        """Enhanced liquidation check with margin calls"""
        for pair in self.currency_pairs:
            if self.positions[pair] != 0:
                margin_level = self.margin_level[pair]
                
                # Check for margin call
                if self.risk_manager.check_margin_call(margin_level, self.margin_call_level):
                    if margin_level > self.liquidation_threshold * 100:
                        # Issue margin call warning
                        self.status_label.config(
                            text=f"⚠️ MARGIN CALL: {pair} - Margin Level: {margin_level:.1f}%"
                        )
                
                # Check for liquidation
                if self.risk_manager.check_liquidation(margin_level):
                    self.force_liquidation(pair)
                        
    def force_liquidation(self, pair):
        """Enhanced force liquidation with detailed logging"""
        if self.positions[pair] != 0:
            liquidation_info = {
                'pair': pair,
                'position_size': self.positions[pair],
                'entry_price': self.entry_prices[pair],
                'liquidation_price': self.current_prices[pair],
                'floating_pnl': self.floating_pnl[pair],
                'margin_level': self.margin_level[pair],
                'timestamp': datetime.now()
            }
            
            self.liquidation_history.append(liquidation_info)
            
            # Execute liquidation
            self.capital[pair] += self.floating_pnl[pair]
            self.positions[pair] = 0
            self.entry_prices[pair] = 0
            self.floating_pnl[pair] = 0
            self.used_margin[pair] = 0
            self.margin_level[pair] = float('inf')
            
            messagebox.showwarning("Force Liquidation", 
                                 f"{pair} position liquidated!\n"
                                 f"Margin Level: {liquidation_info['margin_level']:.1f}%\n"
                                 f"Realized P&L: ${liquidation_info['floating_pnl']:.2f}")
            
            self.status_label.config(text=f"🚨 {pair} LIQUIDATED - Margin Level: {liquidation_info['margin_level']:.1f}%")
            
    def buy_position(self):
        """Enhanced buy position with dynamic leverage"""
        pair = self.selected_pair.get()
        try:
            amount = float(self.trade_amount.get())
            current_price = self.current_prices[pair]
            
            # Get effective leverage
            account_equity = self.capital[pair] + self.floating_pnl[pair]
            if self.dynamic_leverage_enabled:
                effective_leverage = self.risk_manager.get_dynamic_leverage(
                    account_equity, self.risk_profile
                )
            else:
                effective_leverage = self.leverage
            
            # Calculate required margin
            required_margin = self.risk_manager.calculate_required_margin(
                amount, current_price, effective_leverage
            )
            
            if self.capital[pair] >= required_margin:
                # Update position with weighted average entry price
                if self.positions[pair] == 0:
                    self.entry_prices[pair] = current_price
                else:
                    total_value = self.positions[pair] * self.entry_prices[pair] + amount * current_price
                    total_position = self.positions[pair] + amount
                    self.entry_prices[pair] = total_value / total_position if total_position != 0 else current_price
                    
                self.positions[pair] += amount
                self.capital[pair] -= required_margin
                
                # Log trade
                trade_info = {
                    'pair': pair,
                    'action': 'BUY',
                    'amount': amount,
                    'price': current_price,
                    'leverage': effective_leverage,
                    'margin_used': required_margin,
                    'timestamp': datetime.now()
                }
                self.trade_history.append(trade_info)
                
                self.status_label.config(
                    text=f"✅ Bought {amount} lots of {pair} at {current_price:.4f} (Leverage: {effective_leverage:.1f}x)"
                )
            else:
                messagebox.showerror("Insufficient Funds", 
                                   f"Required margin: ${required_margin:.2f}\n"
                                   f"Available capital: ${self.capital[pair]:.2f}")
                
        except ValueError:
            messagebox.showerror("Input Error", "Please enter a valid trade amount")
            
    def sell_position(self):
        """Enhanced sell position with dynamic leverage"""
        pair = self.selected_pair.get()
        try:
            amount = float(self.trade_amount.get())
            current_price = self.current_prices[pair]
            
            # Get effective leverage
            account_equity = self.capital[pair] + self.floating_pnl[pair]
            if self.dynamic_leverage_enabled:
                effective_leverage = self.risk_manager.get_dynamic_leverage(
                    account_equity, self.risk_profile
                )
            else:
                effective_leverage = self.leverage
            
            # Calculate required margin
            required_margin = self.risk_manager.calculate_required_margin(
                amount, current_price, effective_leverage
            )
            
            if self.capital[pair] >= required_margin:
                # Update position with weighted average entry price
                if self.positions[pair] == 0:
                    self.entry_prices[pair] = current_price
                else:
                    total_value = self.positions[pair] * self.entry_prices[pair] - amount * current_price
                    total_position = self.positions[pair] - amount
                    self.entry_prices[pair] = total_value / total_position if total_position != 0 else current_price
                    
                self.positions[pair] -= amount
                self.capital[pair] -= required_margin
                
                # Log trade
                trade_info = {
                    'pair': pair,
                    'action': 'SELL',
                    'amount': amount,
                    'price': current_price,
                    'leverage': effective_leverage,
                    'margin_used': required_margin,
                    'timestamp': datetime.now()
                }
                self.trade_history.append(trade_info)
                
                self.status_label.config(
                    text=f"✅ Sold {amount} lots of {pair} at {current_price:.4f} (Leverage: {effective_leverage:.1f}x)"
                )
            else:
                messagebox.showerror("Insufficient Funds", 
                                   f"Required margin: ${required_margin:.2f}\n"
                                   f"Available capital: ${self.capital[pair]:.2f}")
                
        except ValueError:
            messagebox.showerror("Input Error", "Please enter a valid trade amount")
            
    def close_position(self):
        """Enhanced close position with detailed P&L reporting"""
        pair = self.selected_pair.get()
        
        if self.positions[pair] != 0:
            # Calculate final P&L
            realized_pnl = self.floating_pnl[pair]
            position_size = self.positions[pair]
            entry_price = self.entry_prices[pair]
            exit_price = self.current_prices[pair]
            
            # Update capital
            self.capital[pair] += realized_pnl
            
            # Log trade closure
            trade_info = {
                'pair': pair,
                'action': 'CLOSE',
                'position_size': position_size,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'realized_pnl': realized_pnl,
                'timestamp': datetime.now()
            }
            self.trade_history.append(trade_info)
            
            # Reset position
            self.positions[pair] = 0
            self.entry_prices[pair] = 0
            self.floating_pnl[pair] = 0
            self.used_margin[pair] = 0
            self.margin_level[pair] = float('inf')
            
            self.status_label.config(
                text=f"✅ {pair} position closed - Size: {position_size:.1f}, P&L: ${realized_pnl:.2f}"
            )
        else:
            messagebox.showinfo("No Position", f"No open position for {pair}")
            
    def update_parameters(self):
        """Update enhanced trading parameters"""
        try:
            new_leverage = int(self.leverage_var.get())
            new_margin = float(self.margin_var.get())
            new_liquidation = float(self.liquidation_var.get()) / 100
            new_dynamic_enabled = self.dynamic_leverage_var.get()
            new_risk_profile = self.risk_profile_var.get()
            
            if new_leverage > 0 and new_margin > 0 and 0 < new_liquidation < 1:
                self.leverage = new_leverage
                self.margin_per_lot = new_margin
                self.liquidation_threshold = new_liquidation
                self.dynamic_leverage_enabled = new_dynamic_enabled
                self.risk_profile = new_risk_profile
                
                # Update risk manager
                self.risk_manager.leverage = new_leverage
                self.risk_manager.margin_per_lot = new_margin
                self.risk_manager.liquidation_threshold = new_liquidation
                
                self.status_label.config(text="✅ Advanced trading parameters updated")
                messagebox.showinfo("Parameters Updated", 
                                   f"Leverage: {self.leverage}x\n"
                                   f"Margin: ${self.margin_per_lot}\n"
                                   f"Liquidation: {self.liquidation_threshold*100:.0f}%\n"
                                   f"Dynamic Leverage: {'Enabled' if self.dynamic_leverage_enabled else 'Disabled'}\n"
                                   f"Risk Profile: {self.risk_profile.upper()}")
            else:
                messagebox.showerror("Parameter Error", 
                                   "Invalid parameters:\n"
                                   "- Leverage must be > 0\n"
                                   "- Margin must be > 0\n"
                                   "- Liquidation threshold must be between 0-100%")
                
        except ValueError:
            messagebox.showerror("Parameter Error", "Please enter valid numeric values")

class FXTradingEnvironment:
    """Enhanced trading environment class with advanced risk management"""
    def __init__(self, initial_capital, leverage, margin_per_lot, liquidation_threshold):
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.margin_per_lot = margin_per_lot
        self.liquidation_threshold = liquidation_threshold
        self.risk_manager = AdvancedRiskManager(leverage, margin_per_lot, liquidation_threshold)
        self.reset()
        
    def reset(self):
        """Reset trading environment"""
        self.capital = {'USD/JPY': 1000.0, 'USD/EUR': 1000.0, 'USD/GBP': 1000.0}
        self.positions = {'USD/JPY': 0.0, 'USD/EUR': 0.0, 'USD/GBP': 0.0}
        self.entry_prices = {'USD/JPY': 0.0, 'USD/EUR': 0.0, 'USD/GBP': 0.0}
        self.floating_pnl = {'USD/JPY': 0.0, 'USD/EUR': 0.0, 'USD/GBP': 0.0}
        self.used_margin = {'USD/JPY': 0.0, 'USD/EUR': 0.0, 'USD/GBP': 0.0}

def main():
    """Main function"""
    root = tk.Tk()
    app = ForexTradingGUI(root)
    
    def on_closing():
        if messagebox.askokcancel("Exit", "Are you sure you want to exit the advanced trading system?"):
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
