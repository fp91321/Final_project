###################################### after training model #######################################
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import warnings

# 屏蔽 sklearn MinMaxScaler 的 UserWarning 警告
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.base')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.preprocessing._data')
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')
window_size = 30

class TransformerLSTMTrading:
    def __init__(self, fx_rates, real_fx_rates):
        """
        fx_rates: shape (3, N) → 初始預測匯率，可用 real_fx_rates 的最後一段
        real_fx_rates: shape (3, N) → 所有實際匯率資料
        """
        self.Pre_fx_rates = fx_rates
        self.real_fx_rates = real_fx_rates
        self.day = 0
        self.start = len(fx_rates[0]) - window_size - 1  # 模擬起始點

        self.initial_capital = np.array([1000, 1000, 1000], dtype=float)
        self.capital = np.array([1000, 1000, 1000], dtype=float)
        self.available_margin = self.capital

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

        # ❶ 載入模型與 scaler - 改為 Transformer-LSTM 模型
        self.models = {}
        self.scalers = {}
        
        # 載入三個貨幣對的模型
        currency_pairs = ['USDJPY', 'EURUSD', 'GBPUSD']
        for i, pair in enumerate(currency_pairs):
            try:
                # 載入 Transformer-LSTM 模型
                model = HybridModel(input_dim=11)  # 11個特徵
                model.load_state_dict(torch.load(f"saved_models/{pair}_hybrid_model.pth", map_location='cpu'))
                model.eval()
                self.models[i] = model
                
                # 載入對應的 scaler（如果有的話）
                scaler_path = f"saved_models/{pair}_scaler.pkl"
                if os.path.exists(scaler_path):
                    self.scalers[i] = joblib.load(scaler_path)
                else:
                    # 如果沒有保存的 scaler，創建一個新的
                    self.scalers[i] = MinMaxScaler()
                    
            except Exception as e:
                print(f"載入 {pair} 模型失敗: {e}")
                self.models[i] = None
                self.scalers[i] = MinMaxScaler()

        self.window_size = 60  # Transformer-LSTM 使用 60 天窗口

    def check_liquidation(self, cap_num, maintenance_margin_ratio_threshold=0.3):
        """
        Check if the position should be liquidated (trigger forced liquidation).
        """
        if self.position_size[cap_num] == 0:
            return False
            
        equity = self.capital[cap_num] + self.floating_pnl[cap_num]
        if equity / (self.margin * abs(self.position_size[cap_num])) < maintenance_margin_ratio_threshold:
            return True
        return False

    def close_position(self, cap_num, close_price):
        """
        Close the position and calculate realized PnL.
        """
        return (close_price - self.entry_price[cap_num]) * self.position_size[cap_num] * self.margin * self.leverage[cap_num] / close_price

    def predict_fx_rate(self, data=None):
        """
        使用 Transformer-LSTM 模型預測下一日三種匯率，並更新 self.Pre_fx_rates。
        """
        new_predictions = []
        
        for i in range(3):  # 三個貨幣對
            if self.models[i] is None:
                # 如果模型載入失敗，使用簡單預測
                last_price = self.real_fx_rates[i][self.start + self.day - 1]
                new_predictions.append(last_price * (1 + np.random.normal(0, 0.001)))
                continue
            
            try:
                # ❷ 取得最近 window_size 天的實際匯率
                if self.start + self.day < self.window_size:
                    # 如果數據不足，使用可用的數據
                    start_idx = 0
                    end_idx = self.start + self.day
                else:
                    start_idx = self.start + self.day - self.window_size
                    end_idx = self.start + self.day
                
                # 創建特徵數據（簡化版，只使用價格）
                price_data = self.real_fx_rates[i][start_idx:end_idx]
                
                # 創建模擬的技術指標特徵
                features = self.create_features(price_data)
                
                if len(features) < self.window_size:
                    # 如果特徵數據不足，填充
                    padding = np.tile(features[0], (self.window_size - len(features), 1))
                    features = np.vstack([padding, features])
                elif len(features) > self.window_size:
                    # 如果數據過多，取最後的部分
                    features = features[-self.window_size:]
                
                # ❸ 正規化
                scaled_input = self.scalers[i].fit_transform(features)
                scaled_input = torch.tensor(scaled_input.reshape(1, self.window_size, -1), dtype=torch.float32)
                
                # ❄ 預測
                with torch.no_grad():
                    scaled_pred = self.models[i](scaled_input).cpu().numpy()[0, 0]
                
                # 反正規化（簡化版）
                pred_price = scaled_pred * (np.max(price_data) - np.min(price_data)) + np.min(price_data)
                new_predictions.append(pred_price)
                
            except Exception as e:
                print(f"預測 {i} 號貨幣對失敗: {e}")
                # 使用簡單預測作為備選
                last_price = self.real_fx_rates[i][self.start + self.day - 1]
                new_predictions.append(last_price * (1 + np.random.normal(0, 0.001)))
        
        # ❺ 更新 self.Pre_fx_rates
        new_predictions = np.array(new_predictions).reshape(3, 1)
        self.Pre_fx_rates = np.concatenate([self.Pre_fx_rates, new_predictions], axis=1)

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
        if self.start + self.day >= len(self.Pre_fx_rates[cap_num]):
            return 2, 0  # HOLD if no prediction available
            
        predicted_price = self.Pre_fx_rates[cap_num][self.start + self.day]
        current_price = self.now_price[cap_num]

        price_diff = (predicted_price - current_price) / current_price
        
        if price_diff > 0.002:  # 0.2% threshold
            return 0, 10  # LONG
        elif price_diff < -0.002:
            return 1, 10  # SHORT
        else:
            return 2, 0  # HOLD

    def decide_action(self, cap_num):
        """
        Decide what to do with an existing position.
        """
        if self.start + self.day >= len(self.Pre_fx_rates[cap_num]):
            return 2, 0  # HOLD if no prediction available
            
        predicted_price = self.Pre_fx_rates[cap_num][self.start + self.day]
        current_price = self.now_price[cap_num]
        pos = self.position_size[cap_num]
        pnl = self.floating_pnl[cap_num]

        # 若賺超過 100 或賠超過 -100 就平倉
        if pnl > 100 or pnl < -100:
            return 1, abs(pos)  # CLOSE 全部

        # 若預測方向與持倉方向相反 → 平倉
        price_diff = (predicted_price - current_price) / current_price
        if (pos > 0 and price_diff < -0.002) or (pos < 0 and price_diff > 0.002):
            return 1, abs(pos)  # CLOSE

        # 若預測與方向一致，持有
        return 2, 0  # HOLD

    def update_entry_price(self, cap_num, add_price, old_position, add_position):
        """
        Update the average entry price after adding position.
        """
        old_value = abs(old_position) * self.margin * self.leverage[cap_num]
        add_value = abs(add_position) * self.margin * self.leverage[cap_num]

        if old_value + add_value > 0:
            self.entry_price[cap_num] = (self.entry_price[cap_num] * old_value + add_price * add_value) / (old_value + add_value)

    def update(self):
        """
        Update environment for current day (prices, margins, PnL, position value).
        """
        if self.start + self.day < len(self.real_fx_rates[0]):
            self.now_price = np.array([
                self.real_fx_rates[0][self.start + self.day],
                self.real_fx_rates[1][self.start + self.day],
                self.real_fx_rates[2][self.start + self.day]
            ])

        self.available_margin = self.capital - abs(self.position_size) * self.margin
        self.position_value = abs(self.margin * self.position_size * self.leverage)
        
        # 計算浮動盈虧
        for i in range(3):
            if self.position_size[i] != 0:
                self.floating_pnl[i] = self.position_size[i] * (self.now_price[i] - self.entry_price[i]) * self.leverage[i] * self.margin / self.now_price[i]
            else:
                self.floating_pnl[i] = 0
                
    def simulate_one_day(self, day):
        """模擬一天的交易"""
        self.day = day
        self.update()
        self.predict_fx_rate(None)
        
        # Main trading loop for each currency
        for cap_num in range(3):
            # Check for liquidation first
            if self.position_size[cap_num] != 0 and self.check_liquidation(cap_num):
                print("Liquidation triggered!")
                self.capital[cap_num] += self.close_position(cap_num, self.now_price[cap_num])
                self.position_size[cap_num] = 0
            
            # If no position, try to open new position
            if self.position_size[cap_num] == 0 and self.capital[cap_num] > 0:
                action, num = self.open_position(cap_num, None)
                if action == 0 and num * self.margin <= self.available_margin[cap_num]: # LONG
                    self.position_size[cap_num] += num
                    self.entry_price[cap_num] = self.now_price[cap_num]
                elif action == 1 and num * self.margin <= self.available_margin[cap_num]: # SHORT
                    self.entry_price[cap_num] = self.now_price[cap_num]
                    self.position_size[cap_num] -= num
            else:
                # If position exists, decide what to do
                action, num = self.decide_action(cap_num)
                if action == 0: # ADD position
                    if num * self.margin <= self.available_margin[cap_num]:
                        self.update_entry_price(cap_num, self.now_price[cap_num], self.position_size[cap_num], num)
                        self.position_size[cap_num] += num
                elif action == 1: # CLOSE position
                    self.capital[cap_num] += self.close_position(cap_num, self.now_price[cap_num])
                    self.position_size[cap_num] = 0

    def run_days(self, max_days=None):
        """
        Run simulation over multiple days.
        """
        for day in range(max_days):
            self.day = day
            self.update()
            self.predict_fx_rate(None)

            # Print current state
            print("Day ", day + 1)
            print(" ")

            for i, name in enumerate(["USD/JPY", "USD/EUR", "USD/GBP"]):
                print(name + ":")
                if self.start + day < len(self.Pre_fx_rates[i]):
                    print("Pre_fx_rate: ", self.Pre_fx_rates[i][day + self.start], "real_fx_rates: ", self.now_price[i])
                else:
                    print("Pre_fx_rate: N/A", "real_fx_rates: ", self.now_price[i])
                print("Capital: ", self.capital[i], "available_margin: ", self.available_margin[i], "position_size: ", self.position_size[i], "leverage: ", self.leverage[i])
                print("floating_pnl: ", self.floating_pnl[i], "entry_price: ", self.entry_price[i], "position_value: ", self.position_value[i])
                print(" ")

            # Main trading loop for each currency
            for cap_num in range(3):
                # Check for liquidation
                if self.position_size[cap_num] != 0 and self.check_liquidation(cap_num):
                    print("Liquidation triggered!")
                    self.capital[cap_num] += self.close_position(cap_num, self.now_price[cap_num])
                    self.position_size[cap_num] = 0

                # If no position, try to open new position
                if self.position_size[cap_num] == 0 and self.capital[cap_num] > 0:
                    action, num = self.open_position(cap_num, None)
                    if action == 0 and num * self.margin <= self.available_margin[cap_num]:
                        self.position_size[cap_num] += num
                        self.entry_price[cap_num] = self.now_price[cap_num]
                    elif action == 1 and num * self.margin <= self.available_margin[cap_num]:
                        self.entry_price[cap_num] = self.now_price[cap_num]
                        self.position_size[cap_num] -= num
                else:
                    # If position exists, decide what to do
                    action, num = self.decide_action(cap_num)

                    if action == 0:  # ADD position
                        if num * self.margin <= self.available_margin[cap_num]:
                            self.update_entry_price(cap_num, self.now_price[cap_num], self.position_size[cap_num], num)
                            self.position_size[cap_num] += num
                    elif action == 1:  # CLOSE position
                        self.capital[cap_num] += self.close_position(cap_num, self.now_price[cap_num])
                        self.position_size[cap_num] = 0

        # Final update after run
        for i in range(3):
            if self.position_size[i] != 0:
                self.capital[i] += self.position_size[i] * (self.now_price[i] - self.entry_price[i]) * self.leverage[i] * self.margin / self.now_price[i]
                
        print("Final Results:")
        print("USD/JPY: capital", self.capital[0])
        print("USD/EUR: capital", self.capital[1])
        print("USD/GBP: capital", self.capital[2])
        print("Rate of Return: ", sum(self.capital) / sum(self.initial_capital))


# === Hybrid Transformer-LSTM Model Definition ===
class HybridModel(nn.Module):
    def __init__(self, input_dim, seq_len=60, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lstm = nn.LSTM(d_model, 32, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

"""
# === 執行模擬 ===
# 載入數據
df2 = pd.read_excel('fake_fx_data.xlsx')
real_fx_data = df2.values
real_fx_data = real_fx_data.T

fx_data = real_fx_data[:, -30:]

# fx_rates and real_fx_rates inputs need to be NumPy arrays of shape (3, N)
env = TransformerLSTMTrading(fx_rates=fx_data, real_fx_rates=real_fx_data)
env.run_days(max_days=90)
"""
