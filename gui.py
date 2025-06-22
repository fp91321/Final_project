import tkinter as tk
from tkinter import ttk, messagebox
from tkinter import scrolledtext
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
from keras.models import load_model
import joblib
import os
import torch
import torch.nn as nn
from transformer_lstm import TransformerLSTMTrading, HybridModel

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# block UserWarning 
warnings.filterwarnings('ignore')

# Set matplotlib style for dark theme
plt.style.use('dark_background')
class SimulationWindow:
    def __init__(self, parent, main_gui):
        self.parent = parent
        self.main_gui = main_gui
        self.window = tk.Toplevel(parent)
        
        # 初始化所有必要的屬性
        self.simulation_running = False
        self.fx_trading = None
        self.simulation_thread = None
        
        self.setup_window()
        self.create_widgets()
        
    def setup_window(self):
        """設置模擬視窗"""
        self.window.title("🔮 90-Day Trading Simulation with AI Model")
        self.window.geometry("1600x1000")
        self.window.configure(bg='#1e1e1e')
        self.window.resizable(True, True)
        
        # 設置字體
        self.title_font = ('Arial', 14, 'bold')
        self.header_font = ('Arial', 12, 'bold')
        self.normal_font = ('Arial', 10)
        self.small_font = ('Arial', 9)
        
    def create_widgets(self):
        """創建視窗組件"""
        # 標題
        title_frame = tk.Frame(self.window, bg='#2d2d2d', height=50)
        title_frame.pack(fill='x', padx=10, pady=5)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(title_frame, text="🔮 90-Day AI Trading Simulation Dashboard",
                              font=self.title_font, fg='#00ff88', bg='#2d2d2d')
        title_label.pack(pady=10)
        
        # 主要內容區域
        main_content = tk.Frame(self.window, bg='#1e1e1e')
        main_content.pack(fill='both', expand=True, padx=10, pady=5)
        
        # 左側：控制面板和詳細數據
        left_panel = tk.Frame(main_content, bg='#2d2d2d', width=800)
        left_panel.pack(side='left', fill='both', expand=True, padx=5)
        left_panel.pack_propagate(False)
        
        # 右側：圖表
        right_panel = tk.Frame(main_content, bg='#2d2d2d', width=600)
        right_panel.pack(side='right', fill='both', expand=True, padx=5)
        
        self.create_control_panel(left_panel)
        self.create_detailed_data_panel(left_panel)
        self.create_charts_panel(right_panel)
        
    def create_control_panel(self, parent):
        """創建控制面板"""
        control_frame = tk.LabelFrame(parent, text="🎮 Dual Model Simulation Control",
                                font=self.header_font, fg='#00ff88', bg='#2d2d2d')
        control_frame.pack(fill='x', padx=20, pady=5)
        
        # 數據來源選擇
        data_source_frame = tk.Frame(control_frame, bg='#2d2d2d')
        data_source_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Label(data_source_frame, text="Data Source:", 
                font=self.normal_font, fg='#ffffff', bg='#2d2d2d').pack(side='left')
        
        self.data_source_var = tk.StringVar(value="Excel File")
        data_source_combo = ttk.Combobox(data_source_frame, textvariable=self.data_source_var,
                                        values=['Excel File', 'CSV File', 'Historical Data'], 
                                        state='readonly', width=15)
        data_source_combo.pack(side='left', padx=10)
        
        # 文件路徑選擇
        file_frame = tk.Frame(control_frame, bg='#2d2d2d')
        file_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Label(file_frame, text="Data File:", 
                font=self.normal_font, fg='#ffffff', bg='#2d2d2d').pack(side='left')
        
        self.file_path_var = tk.StringVar(value="fake_fx_data.xlsx")
        file_entry = tk.Entry(file_frame, textvariable=self.file_path_var,
                             font=self.normal_font, width=30)
        file_entry.pack(side='left', padx=5)
        
        browse_btn = tk.Button(file_frame, text="Browse", command=self.browse_file,
                              bg='#0066cc', fg='white', font=self.normal_font)
        browse_btn.pack(side='left', padx=5)
        
        # 控制按鈕
        button_frame = tk.Frame(control_frame, bg='#2d2d2d')
        button_frame.pack(fill='x', padx=10, pady=10)
        
        self.start_btn = tk.Button(button_frame, text="▶️ Start 90-Day Simulation",
                                  command=self.start_simulation,
                                  bg='#00aa00', fg='white', font=self.normal_font, width=25)
        self.start_btn.pack(side='left', padx=5)
        
        self.stop_btn = tk.Button(button_frame, text="⏹️ Stop Simulation",
                                 command=self.stop_simulation,
                                 bg='#aa0000', fg='white', font=self.normal_font, width=20,
                                 state='disabled')
        self.stop_btn.pack(side='left', padx=5)
        
        self.export_btn = tk.Button(button_frame, text="💾 Export Results",
                                   command=self.export_results,
                                   bg='#0066cc', fg='white', font=self.normal_font, width=20)
        self.export_btn.pack(side='left', padx=5)
        
        # 進度條
        progress_frame = tk.Frame(control_frame, bg='#2d2d2d')
        progress_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Label(progress_frame, text="Progress:", 
                font=self.normal_font, fg='#ffffff', bg='#2d2d2d').pack(side='left')
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var,
                                          maximum=90, length=400)
        self.progress_bar.pack(side='left', padx=10)
        
        self.progress_label = tk.Label(progress_frame, text="0/90 days",
                                     font=self.normal_font, fg='#ffffff', bg='#2d2d2d')
        self.progress_label.pack(side='left', padx=10)
        
    def browse_file(self):
        """瀏覽文件"""
        from tkinter import filedialog
        
        data_source = self.data_source_var.get()
        if data_source == "Excel File":
            filetypes = [("Excel files", "*.xlsx *.xls")]
        elif data_source == "CSV File":
            filetypes = [("CSV files", "*.csv")]
        else:
            filetypes = [("All files", "*.*")]
            
        filename = filedialog.askopenfilename(
            title="Select Data File",
            filetypes=filetypes
        )
        
        if filename:
            self.file_path_var.set(filename)
            
    def create_detailed_data_panel(self, parent):
        """創建詳細數據顯示面板"""
        data_frame = tk.LabelFrame(parent, text="📊 Real-time Trading Data", 
                                  font=self.header_font, fg='#00ff88', bg='#2d2d2d')
        data_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # 創建滾動文本區域
        self.data_text = scrolledtext.ScrolledText(
            data_frame, 
            font=self.small_font,
            bg='#1e1e1e', 
            fg='#ffffff',
            insertbackground='white',
            selectbackground='#0066cc',
            wrap=tk.WORD,
            height=25
        )
        self.data_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        # 設置文字顏色標籤
        self.data_text.tag_configure("day_header", foreground="#00ff88", font=('Arial', 11, 'bold'))
        self.data_text.tag_configure("currency_header", foreground="#ffaa00", font=('Arial', 10, 'bold'))
        self.data_text.tag_configure("profit", foreground="#00ff88")
        self.data_text.tag_configure("loss", foreground="#ff6666")
        self.data_text.tag_configure("neutral", foreground="#ffffff")
        self.data_text.tag_configure("final_results", foreground="#00ffff", font=('Arial', 12, 'bold'))
        
    def create_charts_panel(self, parent):
        """創建圖表面板"""
        charts_frame = tk.LabelFrame(parent, text="📈 Trading Charts", 
                                    font=self.header_font, fg='#00ff88', bg='#2d2d2d')
        charts_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # 創建matplotlib圖表
        self.fig = Figure(figsize=(8, 10), facecolor='#2d2d2d')
        self.fig.suptitle('90-Day Trading Simulation', color='white', fontsize=12)
        
        # 三個子圖，每個貨幣對一個
        self.axes = {}
        currency_pairs = ['USD/JPY', 'USD/EUR', 'USD/GBP']
        
        for i, pair in enumerate(currency_pairs):
            ax = self.fig.add_subplot(3, 1, i+1)
            ax.set_facecolor('#1e1e1e')
            ax.tick_params(colors='white', labelsize=8)
            ax.set_ylabel(f'{pair}', color='white', fontsize=9)
            ax.grid(True, alpha=0.3)
            self.axes[pair] = ax
        
        self.canvas = FigureCanvasTkAgg(self.fig, charts_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=5, pady=5)
        
    def load_historical_data(self):
        """載入歷史數據"""
        try:
            file_path = self.file_path_var.get()
            data_source = self.data_source_var.get()
            
            self.data_text.insert(tk.END, f"📂 Loading data from: {file_path}\n", "day_header")
            self.data_text.see(tk.END)
            
            if data_source == "Excel File":
                df = pd.read_excel(file_path)
            elif data_source == "CSV File":
                df = pd.read_csv(file_path)
            else:
                # 嘗試自動檢測文件格式
                if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                    df = pd.read_excel(file_path)
                elif file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                else:
                    raise ValueError("Unsupported file format")
            
            self.data_text.insert(tk.END, f"✅ Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns\n", "profit")
            self.data_text.insert(tk.END, f"📊 Columns: {list(df.columns)}\n", "neutral")
            
            # 檢查必要的列是否存在
            required_columns = ['USDJPY', 'USDEUR', 'USDGBP']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                self.data_text.insert(tk.END, f"❌ Missing columns: {missing_columns}\n", "loss")
                messagebox.showerror("Data Error", f"Missing required columns: {missing_columns}")
                return None
            
            # 轉換為numpy陣列格式 (3, N)
            fx_rates = np.array([
                df['USDJPY'].values,
                df['USDEUR'].values,
                df['USDGBP'].values
            ])
            
            self.data_text.insert(tk.END, f"📈 FX rates shape: {fx_rates.shape}\n", "neutral")
            self.data_text.insert(tk.END, f"📅 Data range: {len(fx_rates[0])} days\n", "neutral")
            
            # 顯示數據預覽
            self.data_text.insert(tk.END, "\n📋 Data Preview (first 5 days):\n", "currency_header")
            for i, pair in enumerate(['USD/JPY', 'USD/EUR', 'USD/GBP']):
                preview_data = fx_rates[i][:5]
                self.data_text.insert(tk.END, f"{pair}: {preview_data}\n", "neutral")
            
            self.data_text.insert(tk.END, "\n", "neutral")
            self.data_text.see(tk.END)
            
            return fx_rates
            
        except FileNotFoundError:
            error_msg = f"File not found: {file_path}"
            self.data_text.insert(tk.END, f"❌ {error_msg}\n", "loss")
            messagebox.showerror("File Error", error_msg)
            return None
            
        except Exception as e:
            error_msg = f"Error loading data: {str(e)}"
            self.data_text.insert(tk.END, f"❌ {error_msg}\n", "loss")
            messagebox.showerror("Data Error", error_msg)
            return None
            
    def start_simulation(self):
        """開始模擬"""
        if self.simulation_running:
            return
            
        # 首先載入歷史數據
        fx_rates = self.load_historical_data()
        if fx_rates is None:
            return
            
        # 檢查數據長度是否足夠
        if fx_rates.shape[1] < 120:  # 至少需要120天數據
            messagebox.showerror("Data Error", 
                               f"Insufficient data. Need at least 120 days, got {fx_rates.shape[1]} days")
            return
            
        self.simulation_running = True
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        
        # 清空之前的模擬數據
        self.data_text.insert(tk.END, "\n🚀 Starting 90-Day AI Trading Simulation...\n\n", "day_header")
        
        # 重置進度
        self.progress_var.set(0)
        self.progress_label.config(text="0/90 days")
        
        # 儲存歷史數據供模擬使用
        self.historical_fx_rates = fx_rates
        
        # 在新線程中運行模擬
        self.simulation_thread = threading.Thread(target=self.run_simulation)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
        
    def stop_simulation(self):
        """停止模擬"""
        self.simulation_running = False
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        
    def run_simulation(self):
        """運行90天模擬"""
        try:
            # 使用載入的歷史數據創建FXTrading實例
            self.fx_trading = self.create_fx_trading_instance()
            
            # 運行90天模擬
            for day in range(90):
                if not self.simulation_running:
                    break
                    
                # 更新模擬一天
                self.simulate_one_day(day)
                
                # 每天都顯示詳細數據
                self.window.after(0, self.display_day_data, day + 1)
                
                # 更新進度
                self.window.after(0, self.update_progress, day + 1)
                
                # 更新圖表（每5天更新一次以提高性能）
                if day % 5 == 0 or day == 89:
                    self.window.after(0, self.update_charts)
                
                # 模擬延遲
                time.sleep(0.05)
                
            # 顯示最終結果
            if self.simulation_running:
                self.window.after(0, self.display_final_results)
                
        except Exception as e:
            error_msg = str(e)  # 立即獲取錯誤訊息
            self.window.after(0, lambda: messagebox.showerror("Simulation Error", f"模擬過程中發生錯誤: {error_msg}"))
        finally:
            self.simulation_running = False
            self.window.after(0, lambda: self.start_btn.config(state='normal'))
            self.window.after(0, lambda: self.stop_btn.config(state='disabled'))
            
    def create_fx_trading_instance(self):
        """使用歷史數據創建FXTrading實例"""
        try:
            # 使用前30天作為初始預測數據，後90天作為模擬數據
            initial_rates = self.historical_fx_rates[:, :30]  # 前30天作為初始預測
            
            # 創建FXTrading實例（會自動載入GRU模型）
            fx_trading = FXTrading(initial_rates, self.historical_fx_rates)
            
            return fx_trading
        except Exception as e:
            messagebox.showerror("Model Loading Error", 
                            f"無法載入GRU模型:\n{str(e)}\n\n請確保以下檔案存在於程式目錄中:\n- fx_model_gru.h5\n- scaler.pkl")
            return None

        
    def simulate_one_day(self, day):
        """模擬一天的交易 - 與final.ipynb完全一致"""
        self.fx_trading.day = day
        self.fx_trading.update()
        self.fx_trading.predict_fx_rate(None)

        # Main trading loop for each currency
        for cap_num in range(3):
            # Check for liquidation first
            if self.fx_trading.position_size[cap_num] != 0 and self.fx_trading.check_liquidation(cap_num):
                print("Liquidation triggered!")
                self.fx_trading.capital[cap_num] += self.fx_trading.close_position(cap_num, self.fx_trading.now_price[cap_num])
                self.fx_trading.position_size[cap_num] = 0

            # If no position, try to open new position
            if self.fx_trading.position_size[cap_num] == 0 and self.fx_trading.capital[cap_num] > 0:
                action, num = self.fx_trading.open_position(cap_num, None)
                if action == 0 and num * self.fx_trading.margin <= self.fx_trading.available_margin[cap_num]:  # LONG
                    self.fx_trading.position_size[cap_num] = num
                    self.fx_trading.entry_price[cap_num] = self.fx_trading.now_price[cap_num]
                elif action == 1 and num * self.fx_trading.margin <= self.fx_trading.available_margin[cap_num]:  # SHORT
                    self.fx_trading.position_size[cap_num] = -num
                    self.fx_trading.entry_price[cap_num] = self.fx_trading.now_price[cap_num]
            else:
                # If position exists, decide what to do
                action, num = self.fx_trading.decide_action(None)

                if action == 0:  # ADD position
                    if self.fx_trading.position_size[cap_num] > 0:  # 多頭加倉
                        self.fx_trading.update_entry_price(cap_num, self.fx_trading.now_price[cap_num], self.fx_trading.position_size[cap_num], num)
                        self.fx_trading.position_size[cap_num] += num
                    else:  # 空頭加倉
                        self.fx_trading.update_entry_price(cap_num, self.fx_trading.now_price[cap_num], self.fx_trading.position_size[cap_num], -num)
                        self.fx_trading.position_size[cap_num] -= num
                elif action == 1:  # CLOSE position
                    self.fx_trading.capital[cap_num] += self.fx_trading.close_position(cap_num, self.fx_trading.now_price[cap_num])
                    self.fx_trading.position_size[cap_num] = 0

        # Transformer-LSTM model trading
        if self.fx_trading.transformer_lstm_trading:
            self.fx_trading.transformer_lstm_trading.day = day
            self.fx_trading.transformer_lstm_trading.update()
            self.fx_trading.transformer_lstm_trading.predict_fx_rate(None)
            
            # Transformer-LSTM trading logic
            for cap_num in range(3):
                # Check for liquidation first
                if self.fx_trading.transformer_lstm_trading.position_size[cap_num] != 0 and self.fx_trading.transformer_lstm_trading.check_liquidation(cap_num):
                    print("Transformer-LSTM Liquidation triggered!")
                    self.fx_trading.transformer_lstm_trading.capital[cap_num] += self.fx_trading.transformer_lstm_trading.close_position(cap_num, self.fx_trading.transformer_lstm_trading.now_price[cap_num])
                    self.fx_trading.transformer_lstm_trading.position_size[cap_num] = 0
                
                # Trading decision logic (similar to GRU)
                if self.fx_trading.transformer_lstm_trading.position_size[cap_num] == 0 and self.fx_trading.transformer_lstm_trading.capital[cap_num] > 0:
                    action, num = self.fx_trading.transformer_lstm_trading.open_position(cap_num, None)
                    if action == 0 and num * self.fx_trading.transformer_lstm_trading.margin <= self.fx_trading.transformer_lstm_trading.available_margin[cap_num]:
                        self.fx_trading.transformer_lstm_trading.position_size[cap_num] = num
                        self.fx_trading.transformer_lstm_trading.entry_price[cap_num] = self.fx_trading.transformer_lstm_trading.now_price[cap_num]
                    elif action == 1 and num * self.fx_trading.transformer_lstm_trading.margin <= self.fx_trading.transformer_lstm_trading.available_margin[cap_num]:
                        self.fx_trading.transformer_lstm_trading.position_size[cap_num] = -num
                        self.fx_trading.transformer_lstm_trading.entry_price[cap_num] = self.fx_trading.transformer_lstm_trading.now_price[cap_num]
                else:
                    action, num = self.fx_trading.transformer_lstm_trading.decide_action(cap_num)
                    if action == 1:  # CLOSE position
                        self.fx_trading.transformer_lstm_trading.capital[cap_num] += self.fx_trading.transformer_lstm_trading.close_position(cap_num, self.fx_trading.transformer_lstm_trading.now_price[cap_num])
                        self.fx_trading.transformer_lstm_trading.position_size[cap_num] = 0        


    def display_day_data(self, day):
        """Display daily detailed data - including dual model comparison in English"""
        currency_pairs = ['USD/JPY', 'USD/EUR', 'USD/GBP']
        
        self.data_text.insert(tk.END, f"Day {day}\n", "day_header")
        self.data_text.insert(tk.END, "="*50 + "\n\n", "neutral")
        
        # GRU Model Results
        self.data_text.insert(tk.END, "GRU Model:\n", "currency_header")
        for i, pair in enumerate(currency_pairs):
            self.data_text.insert(tk.END, f"{pair}:\n", "currency_header")
            
            # Real rate
            real_rate = self.fx_trading.now_price[i]
            self.data_text.insert(tk.END, f"real_fx_rates: {real_rate:.6f}\n", "neutral")
            
            # GRU prediction
            if len(self.fx_trading.predictions['gru']) > 0:
                gru_pred = self.fx_trading.predictions['gru'][-1][i]
                self.data_text.insert(tk.END, f"Pre_fx_rate: {gru_pred:.6f}\n", "profit")
            
            # Trading info
            capital = self.fx_trading.capital[i]
            available_margin = self.fx_trading.available_margin[i]
            position_size = self.fx_trading.position_size[i]
            leverage = self.fx_trading.leverage[i]
            floating_pnl = self.fx_trading.floating_pnl[i]
            entry_price = self.fx_trading.entry_price[i]
            position_value = abs(position_size) * self.fx_trading.margin * leverage
            
            self.data_text.insert(tk.END, f"Capital: {capital:.13f} available_margin: {available_margin:.13f} position_size: {position_size:.1f} leverage: {leverage}\n", "neutral")
            self.data_text.insert(tk.END, f"floating_pnl: {floating_pnl:.13f} entry_price: {entry_price:.13f} position_value: {position_value:.1f}\n", "neutral")
            self.data_text.insert(tk.END, "\n", "neutral")
        
        # Transformer-LSTM Model Results
        self.data_text.insert(tk.END, "Transformer-LSTM Model:\n", "currency_header")
        for i, pair in enumerate(currency_pairs):
            self.data_text.insert(tk.END, f"{pair}:\n", "currency_header")
            
            # Real rate
            real_rate = self.fx_trading.now_price[i]
            self.data_text.insert(tk.END, f"real_fx_rates: {real_rate:.6f}\n", "neutral")
            
            # Transformer-LSTM prediction
            if len(self.fx_trading.predictions['transformer_lstm']) > 0:
                transformer_pred = self.fx_trading.predictions['transformer_lstm'][-1][i]
                self.data_text.insert(tk.END, f"Pre_fx_rate: {transformer_pred:.6f}\n", "profit")
            
            # Trading info for Transformer-LSTM
            if self.fx_trading.transformer_lstm_trading:
                capital = self.fx_trading.transformer_lstm_trading.capital[i]
                available_margin = self.fx_trading.transformer_lstm_trading.available_margin[i]
                position_size = self.fx_trading.transformer_lstm_trading.position_size[i]
                leverage = self.fx_trading.transformer_lstm_trading.leverage[i]
                floating_pnl = self.fx_trading.transformer_lstm_trading.floating_pnl[i]
                entry_price = self.fx_trading.transformer_lstm_trading.entry_price[i]
                position_value = abs(position_size) * self.fx_trading.transformer_lstm_trading.margin * leverage
                
                self.data_text.insert(tk.END, f"Capital: {capital:.13f} available_margin: {available_margin:.13f} position_size: {position_size:.1f} leverage: {leverage}\n", "neutral")
                self.data_text.insert(tk.END, f"floating_pnl: {floating_pnl:.13f} entry_price: {entry_price:.13f} position_value: {position_value:.1f}\n", "neutral")
            
            self.data_text.insert(tk.END, "\n", "neutral")
        
        self.data_text.see(tk.END)

        
    def display_final_results(self):
        """Display final results - dual model comparison in English"""
        currency_pairs = ['USD/JPY', 'USD/EUR', 'USD/GBP']
        
        # GRU Model Final Results
        self.data_text.insert(tk.END, "\n" + "="*60 + "\n", "final_results")
        self.data_text.insert(tk.END, "GRU Model Final Results:\n", "final_results")
        
        gru_total = 0
        for i, pair in enumerate(currency_pairs):
            capital = self.fx_trading.capital[i]
            gru_total += capital
            self.data_text.insert(tk.END, f"{pair}: capital {capital:.13f}\n", "profit")
        
        gru_return = gru_total / 3000
        self.data_text.insert(tk.END, f"Rate of Return: {gru_return:.13f}\n", "final_results")
        
        # Transformer-LSTM Model Final Results
        self.data_text.insert(tk.END, "\nTransformer-LSTM Model Final Results:\n", "final_results")
        
        transformer_total = 0
        if self.fx_trading.transformer_lstm_trading:
            for i, pair in enumerate(currency_pairs):
                capital = self.fx_trading.transformer_lstm_trading.capital[i]
                transformer_total += capital
                self.data_text.insert(tk.END, f"{pair}: capital {capital:.13f}\n", "profit")
            
            transformer_return = transformer_total / 3000
            self.data_text.insert(tk.END, f"Rate of Return: {transformer_return:.13f}\n", "final_results")
            
            # Model Comparison
            self.data_text.insert(tk.END, "\n" + "="*60 + "\n", "final_results")
            self.data_text.insert(tk.END, "Model Performance Comparison:\n", "final_results")
            
            if gru_return > transformer_return:
                winner = "GRU"
                difference = gru_return - transformer_return
                self.data_text.insert(tk.END, f"🏆 {winner} Model Wins!\n", "profit")
            else:
                winner = "Transformer-LSTM"
                difference = transformer_return - gru_return
                self.data_text.insert(tk.END, f"🏆 {winner} Model Wins!\n", "profit")
            
            self.data_text.insert(tk.END, f"Return Rate Difference: {difference:.13f}\n", "neutral")
        
        self.data_text.insert(tk.END, "="*60 + "\n", "final_results")
        
    def update_progress(self, day):
        """更新進度條"""
        self.progress_var.set(day)
        self.progress_label.config(text=f"{day}/90 days")
        
    def update_charts(self):
        """更新圖表 - 顯示兩個模型預測比較"""
        if not self.fx_trading:
            return

        currency_pairs = ['USD/JPY', 'USD/EUR', 'USD/GBP']
        
        for i, pair in enumerate(currency_pairs):
            ax = self.axes[pair]
            ax.clear()
            
            current_day = self.fx_trading.day
            if current_day > 0:
                days = list(range(1, current_day + 1))
                actual_prices = []
                gru_predictions = []
                transformer_predictions = []
                
                for day in range(current_day):
                    actual_prices.append(self.fx_trading.real_fx_rates[i][self.fx_trading.start + day])
                    
                    if day < len(self.fx_trading.predictions['gru']):
                        gru_predictions.append(self.fx_trading.predictions['gru'][day][i])
                    if day < len(self.fx_trading.predictions['transformer_lstm']):
                        transformer_predictions.append(self.fx_trading.predictions['transformer_lstm'][day][i])
                
                # 繪製實際價格 - 確保有 label
                if actual_prices:
                    ax.plot(days, actual_prices, color='#00ff88', linewidth=2, label='Actual')
                
                # 繪製GRU預測 - 確保有 label
                if gru_predictions and len(gru_predictions) == len(days):
                    ax.plot(days, gru_predictions, color='#ffaa00', linewidth=2, linestyle='--', label='GRU predict')
                
                # 繪製Transformer-LSTM預測 - 確保有 label
                if transformer_predictions and len(transformer_predictions) == len(days):
                    ax.plot(days, transformer_predictions, color='#ff6600', linewidth=2, linestyle=':', label='Transformer-LSTM predict')
            
            # 設置圖表樣式
            ax.set_facecolor('#1e1e1e')
            ax.tick_params(colors='white', labelsize=8)
            ax.set_ylabel(f'{pair}', color='white', fontsize=9)
            ax.grid(True, alpha=0.3)
            
            # 只有在有數據時才顯示圖例
            handles, labels = ax.get_legend_handles_labels()
            if handles:  # 確保有可用的 handles
                ax.legend(fontsize=8)
        
        self.fig.tight_layout()
        self.canvas.draw()

        
    def export_results(self):
        """匯出結果"""
        if not self.fx_trading:
            messagebox.showwarning("No Data", "沒有可匯出的模擬數據")
            return
            
        try:
            # 獲取文本內容
            content = self.data_text.get(1.0, tk.END)
            
            # 儲存為文本文件
            filename = f"simulation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            
            messagebox.showinfo("Export Success", f"結果已匯出至: {filename}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"匯出失敗: {str(e)}")

# 需要在final.ipynb中的FXTrading類基礎上進行的修改
class FXTrading:
    def __init__(self, fx_rates, real_fx_rates):
        """
        fx_rates: shape (3, N) → 初始預測匯率
        real_fx_rates: shape (3, N) → 所有實際匯率資料
        """
        self.Pre_fx_rates = fx_rates.copy()
        self.real_fx_rates = real_fx_rates
        self.day = 0
        self.start = len(fx_rates[0]) - 30 - 1  # 模擬起始點

        self.initial_capital = np.array([1000, 1000, 1000], dtype=float)
        self.capital = np.array([1000, 1000, 1000], dtype=float)
        self.available_margin = self.capital.copy()

        self.leverage = np.array([5, 5, 5])
        self.position_size = np.array([0, 0, 0], dtype=float)
        self.position_value = np.array([0, 0, 0], dtype=float)
        self.floating_pnl = np.array([0, 0, 0], dtype=float)

        self.now_price = np.array([
            real_fx_rates[0][self.start],
            real_fx_rates[1][self.start],
            real_fx_rates[2][self.start]
        ], dtype=float)

        self.entry_price = np.array([0, 0, 0], dtype=float)
        self.margin = 10
        self.window_size = 30  # 新增：GRU模型需要的時間窗口

        # 新增：載入兩個模型
        self.models = {}
        self.scalers = {}
        
        # 載入GRU模型
        try:
            self.models['gru'] = load_model("fx_model_gru.h5", compile=False)
            self.scalers['gru'] = joblib.load("scaler.pkl")
            print("✅ 成功載入GRU模型")
        except Exception as e:
            print(f"❌ 載入GRU模型失敗: {e}")
            self.models['gru'] = None
            self.scalers['gru'] = None
            
        # 載入Transformer-LSTM模型
        try:
            import torch
            from transformer_lstm import HybridModel
            
            self.models['transformer_lstm'] = {}
            self.scalers['transformer_lstm'] = {}
            
            currency_pairs = ['USDJPY', 'EURUSD', 'GBPUSD']
            for i, pair in enumerate(currency_pairs):
                model = HybridModel(input_dim=11)
                model.load_state_dict(torch.load(f"saved_models/{pair}_hybrid_model.pth", map_location='cpu'))
                model.eval()
                self.models['transformer_lstm'][i] = model
                
                scaler_path = f"saved_models/{pair}_scaler.pkl"
                if os.path.exists(scaler_path):
                    self.scalers['transformer_lstm'][i] = joblib.load(scaler_path)
                else:
                    from sklearn.preprocessing import MinMaxScaler
                    self.scalers['transformer_lstm'][i] = MinMaxScaler()
            print("✅ 成功載入Transformer-LSTM模型")
        except Exception as e:
            print(f"❌ 載入Transformer-LSTM模型失敗: {e}")
            self.models['transformer_lstm'] = {}
            self.scalers['transformer_lstm'] = {}
        
        # 預測結果儲存
        self.predictions = {
            'gru': [],
            'transformer_lstm': []
        }
        
        # 新增：初始化 Transformer-LSTM 交易環境
        try:
            from transformer_lstm import TransformerLSTMTrading
            self.transformer_lstm_trading = TransformerLSTMTrading(fx_rates, real_fx_rates)
        except Exception as e:
            print(f"❌ 初始化Transformer-LSTM交易環境失敗: {e}")
            self.transformer_lstm_trading = None
        
        # 儲存兩個模型的交易結果
        self.trading_results = {
            'gru': {'capital': np.array([1000, 1000, 1000], dtype=float)},
            'transformer_lstm': {'capital': np.array([1000, 1000, 1000], dtype=float)}
        }
        
    def predict_with_transformer_lstm(self):
        """Transformer-LSTM模型預測"""
        predictions = []
        
        for i in range(3):
            if i not in self.models['transformer_lstm']:
                predictions.append(self.real_fx_rates[i][self.start + self.day - 1])
                continue
                
            try:
                # 取得歷史數據
                if self.start + self.day < 60:
                    start_idx = 0
                    end_idx = self.start + self.day
                else:
                    start_idx = self.start + self.day - 60
                    end_idx = self.start + self.day
                
                price_data = self.real_fx_rates[i][start_idx:end_idx]
                features = self.create_features(price_data)
                
                if len(features) < 60:
                    padding = np.tile(features[0], (60 - len(features), 1))
                    features = np.vstack([padding, features])
                elif len(features) > 60:
                    features = features[-60:]
                
                scaled_input = self.scalers['transformer_lstm'][i].fit_transform(features)
                scaled_input = torch.tensor(scaled_input.reshape(1, 60, -1), dtype=torch.float32)
                
                with torch.no_grad():
                    scaled_pred = self.models['transformer_lstm'][i](scaled_input).cpu().numpy()[0, 0]
                
                pred_price = scaled_pred * (np.max(price_data) - np.min(price_data)) + np.min(price_data)
                predictions.append(pred_price)
                
            except Exception as e:
                print(f"Transformer-LSTM預測失敗: {e}")
                predictions.append(self.real_fx_rates[i][self.start + self.day - 1])
        
        return np.array(predictions)

    def predict_fx_rate(self, data=None):
        """同時使用兩個模型進行預測"""
        gru_predictions = self.predict_with_gru()
        transformer_predictions = self.predict_with_transformer_lstm()
        
        # 儲存預測結果用於比較
        self.predictions['gru'].append(gru_predictions)
        self.predictions['transformer_lstm'].append(transformer_predictions)
        
        # 使用GRU預測作為主要交易信號（或可以改為平均）
        new_predictions = gru_predictions.reshape(3, 1)
        self.Pre_fx_rates = np.concatenate([self.Pre_fx_rates, new_predictions], axis=1)
        
    def predict_with_gru(self):
        """GRU模型預測"""
        recent_days = []
        for i in range(self.start + self.day - self.window_size, self.start + self.day):
            recent_days.append([
                self.real_fx_rates[0][i],
                self.real_fx_rates[1][i],
                self.real_fx_rates[2][i]
            ])
        
        recent_days = np.array(recent_days)
        scaled_input = self.scalers['gru'].transform(recent_days)
        scaled_input = scaled_input.reshape(1, self.window_size, 3)
        
        scaled_pred = self.models['gru'].predict(scaled_input, verbose=0)[0]
        real_pred = self.scalers['gru'].inverse_transform([scaled_pred])[0]
        
        return real_pred
    
    def predict_fx_rate(self, data=None):
        """同時使用兩個模型進行預測"""
        gru_predictions = self.predict_with_gru()
        transformer_predictions = self.predict_with_transformer_lstm()
        
        # 儲存預測結果用於比較
        self.predictions['gru'].append(gru_predictions)
        self.predictions['transformer_lstm'].append(transformer_predictions)
        
        # 使用GRU預測作為主要交易信號（或可以改為平均）
        new_predictions = gru_predictions.reshape(3, 1)
        self.Pre_fx_rates = np.concatenate([self.Pre_fx_rates, new_predictions], axis=1)
        
    def predict_with_gru(self):
        """GRU模型預測"""
        recent_days = []
        for i in range(self.start + self.day - self.window_size, self.start + self.day):
            recent_days.append([
                self.real_fx_rates[0][i],
                self.real_fx_rates[1][i],
                self.real_fx_rates[2][i]
            ])
        
        recent_days = np.array(recent_days)
        scaled_input = self.scalers['gru'].transform(recent_days)
        scaled_input = scaled_input.reshape(1, self.window_size, 3)
        
        scaled_pred = self.models['gru'].predict(scaled_input, verbose=0)[0]
        real_pred = self.scalers['gru'].inverse_transform([scaled_pred])[0]
        
        return real_pred
    
    def create_features(self, price_data):
        """
        從價格數據創建技術指標特徵
        """
        if len(price_data) < 50:
            # 如果數據不足，創建簡化特徵
            features = np.zeros((len(price_data), 11))
            features[:, 0] = price_data  # Close price
            for i in range(1, 11):
                features[:, i] = price_data  # 其他特徵暫時用價格填充
            return features
        
        df = pd.DataFrame({
            'Close': price_data,
            'High': price_data,
            'Low': price_data,
            'Open': price_data
        })
        
        # 計算技術指標
        df['MA10'] = df['Close'].rolling(10, min_periods=1).mean()
        df['MA50'] = df['Close'].rolling(50, min_periods=1).mean()
        df['EMA10'] = df['Close'].ewm(span=10).mean()
        df['EMA50'] = df['Close'].ewm(span=50).mean()
        df['STD20'] = df['Close'].rolling(20, min_periods=1).std()
        df['Bollinger_High'] = df['MA10'] + 2 * df['STD20']
        df['Bollinger_Low'] = df['MA10'] - 2 * df['STD20']
        df['Bollinger_Width'] = df['Bollinger_High'] - df['Bollinger_Low']
        
        # RSI simplify
        pct_change = df['Close'].pct_change()
        df['RSI'] = 50 + pct_change.rolling(14, min_periods=1).mean() * 100
        
        df['MOM'] = df['Close'].diff(10).fillna(0)
        df['MACD'] = df['EMA10'] - df['EMA50']
        df['ATR'] = df['Close'].diff().abs().rolling(14, min_periods=1).mean()
        
        # 選擇特徵
        features = ['Close', 'MA10', 'MA50', 'EMA10', 'EMA50', 'STD20',
                   'Bollinger_Width', 'RSI', 'MOM', 'MACD', 'ATR']
        
        return df[features].ffill().bfill().values
    
    def open_position(self, cap_num, any):
        """
        Decide how to open a position based on predicted vs. current price.
        """
        predicted_price = self.Pre_fx_rates[cap_num][self.start + self.day]
        current_price = self.now_price[cap_num]

        if predicted_price > current_price:
            return 0, 10  # LONG
        elif predicted_price < current_price:
            return 1, 10  # SHORT
        else:
            return 2, 0  # HOLD

    def decide_action(self, any):
        """
        Decide what to do with an existing position.
        """
        actions = []

        for cap_num in range(3):
            predicted_price = self.Pre_fx_rates[cap_num][self.start + self.day]
            current_price = self.now_price[cap_num]
            pos = self.position_size[cap_num]
            pnl = self.floating_pnl[cap_num]

            # 若賺超過 100 或賠超過 -100 就平倉
            if pnl > 100 or pnl < -100:
                actions.append((1, abs(pos)))  # CLOSE 全部
                continue

            # 若預測方向與持倉方向相反 → 平倉
            if (pos > 0 and predicted_price < current_price) or (pos < 0 and predicted_price > current_price):
                actions.append((1, abs(pos)))  # CLOSE
                continue

            # 若預測與方向一致，可選擇加碼
            actions.append((2, 0))  # HOLD

        return actions[self.day % 3]  # 每次 call 只會用一個 cap_num，所以這樣 round-robin 給個值


    def check_liquidation(self, cap_num, maintenance_margin_ratio_threshold=0.3):
        """
        Check if the position should be liquidated (trigger forced liquidation).
        """
        if self.position_size[cap_num] == 0:
            return False
            
        equity = self.capital[cap_num] + self.floating_pnl[cap_num]
        used_margin = self.margin * abs(self.position_size[cap_num])
        
        if used_margin == 0:
            return False
            
        return equity / used_margin < maintenance_margin_ratio_threshold


    def close_position(self, cap_num, close_price):
        """平倉並計算實現損益"""
        if self.position_size[cap_num] == 0 or self.entry_price[cap_num] == 0:
            return 0
        return (close_price - self.entry_price[cap_num]) * self.position_size[cap_num] * self.margin * self.leverage[cap_num] / close_price

    def update_entry_price(self, cap_num, add_price, old_position, add_position):
        """更新平均進場價格"""
        if old_position == 0:
            self.entry_price[cap_num] = add_price
            return
            
        old_value = abs(old_position) * self.margin * self.leverage[cap_num]
        add_value = abs(add_position) * self.margin * self.leverage[cap_num]
        
        if old_value + add_value > 0:
            self.entry_price[cap_num] = (self.entry_price[cap_num] * old_value + add_price * add_value) / (old_value + add_value)

    def update(self):
        """
        Update environment for current day (prices, margins, PnL, position value).
        """
        self.now_price = np.array([
            self.real_fx_rates[0][self.start + self.day],
            self.real_fx_rates[1][self.start + self.day],
            self.real_fx_rates[2][self.start + self.day]
        ])

        self.available_margin = self.capital - abs(self.position_size) * self.margin
        self.position_value = abs(self.margin * self.position_size * self.leverage)
        self.floating_pnl = self.position_size * (self.now_price - self.entry_price) * self.leverage * self.margin / self.now_price



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

        # 新增90天模擬按鈕
        simulate_btn = tk.Button(button_frame, text="🔮 90-Day Simulation", 
                            command=self.open_simulation_window,
                            bg='#9900cc', fg='white', font=self.header_font, width=30)
        simulate_btn.pack(side='left', padx=5)

    def open_simulation_window(self):
        """開啟90天模擬視窗"""
        simulation_window = SimulationWindow(self.root, self)

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
