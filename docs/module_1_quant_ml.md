# 📊 Module 1: Quant ML Pipeline — XGBoost Time-Series Forecasting

> **Module:** Quantitative Machine Learning  
> **Phiên bản:** v1.0 | **Ngày:** 2026-05-06

---

## 1. Mục tiêu Module

Dự báo **xác suất xu hướng VN-Index ngày mai** (Tăng / Không tăng) dựa trên dữ liệu OHLCV lịch sử và các chỉ báo kỹ thuật. Module này hoạt động **hoàn toàn độc lập** với LLM — đầu ra là một vector xác suất `{P_up, P_down}` được truyền vào LangGraph State.

---

## 2. Luồng Xử lý Tổng thể

```
[yfinance] → Raw OHLCV DataFrame
        │
        ▼
[Data Cleaner] → Xử lý missing values, forward-fill
        │
        ▼
[Feature Engineer] → Tính 20+ Technical Indicators
        │
        ▼
[Label Generator] → Tạo target variable (binary: 0/1)
        │
        ▼
[Train/Test Split] → TimeSeriesSplit (không dùng random shuffle)
        │
        ▼
[XGBoost Classifier] → Training với early stopping
        │
        ▼
[Model Serializer] → Lưu .pkl + metadata .json
        │
        ▼
[Predictor] → predict_proba() → P(Tăng), P(Giảm)
```

---

## 3. Data Ingestion — `scraper/market_data.py`

### 3.1 Nguồn Dữ liệu

```
Ticker:   ^VNINDEX (Yahoo Finance)
Period:   504 ngày giao dịch (~2 năm)
Interval: 1 ngày (daily OHLCV)
Fields:   Open, High, Low, Close, Volume
```

### 3.2 Data Schema

| Cột | Kiểu | Mô tả |
|-----|------|--------|
| `date` | datetime | Ngày giao dịch (index) |
| `open` | float | Giá mở cửa |
| `high` | float | Giá cao nhất |
| `low` | float | Giá thấp nhất |
| `close` | float | Giá đóng cửa |
| `volume` | int | Khối lượng giao dịch |

### 3.3 Data Quality Rules

- Loại bỏ hàng có `close = 0` hoặc `volume = 0`
- `forward_fill` cho ngày nghỉ lễ
- Đảm bảo index là `DatetimeIndex` dạng UTC

---

## 4. Feature Engineering — `ml_models/feature_engineer.py`

Sử dụng thư viện `ta` (Technical Analysis Library) để tạo features. Tất cả features được tính từ OHLCV raw — **không dùng future data** để tránh data leakage.

### 4.1 Bảng Features Đầy đủ

| Nhóm | Feature | Thư viện | Tham số | Ý nghĩa |
|------|---------|----------|---------|---------|
| **Momentum** | RSI_14 | `ta.momentum.RSIIndicator` | window=14 | Chỉ số sức mạnh tương đối |
| **Momentum** | Stoch_K | `ta.momentum.StochasticOscillator` | k=14, d=3 | Stochastic K |
| **Momentum** | Stoch_D | `ta.momentum.StochasticOscillator` | k=14, d=3 | Stochastic D |
| **Trend** | MACD | `ta.trend.MACD` | fast=12, slow=26 | MACD Line |
| **Trend** | MACD_signal | `ta.trend.MACD` | signal=9 | Signal Line |
| **Trend** | MACD_diff | `ta.trend.MACD` | — | MACD Histogram |
| **Trend** | EMA_9 | `ta.trend.EMAIndicator` | window=9 | EMA 9 ngày |
| **Trend** | EMA_21 | `ta.trend.EMAIndicator` | window=21 | EMA 21 ngày |
| **Trend** | SMA_50 | `ta.trend.SMAIndicator` | window=50 | SMA 50 ngày |
| **Trend** | ADX | `ta.trend.ADXIndicator` | window=14 | Average Directional Index |
| **Volatility** | BB_upper | `ta.volatility.BollingerBands` | window=20 | Bollinger Band trên |
| **Volatility** | BB_lower | `ta.volatility.BollingerBands` | window=20 | Bollinger Band dưới |
| **Volatility** | BB_width | Tính tay | — | `(upper-lower)/middle` |
| **Volatility** | ATR_14 | `ta.volatility.AverageTrueRange` | window=14 | Average True Range |
| **Volume** | OBV | `ta.volume.OnBalanceVolumeIndicator` | — | On-Balance Volume |
| **Volume** | VMA_20 | Rolling mean(volume) | window=20 | Volume Moving Average |
| **Volume** | Volume_ratio | Tính tay | — | `volume / VMA_20` |
| **Price Action** | Return_1d | `close.pct_change(1)` | — | Lợi nhuận 1 ngày |
| **Price Action** | Return_5d | `close.pct_change(5)` | — | Lợi nhuận 5 ngày |
| **Price Action** | Return_20d | `close.pct_change(20)` | — | Lợi nhuận 20 ngày |
| **Price Action** | High_Low_pct | Tính tay | — | `(high-low)/close` |
| **Calendar** | day_of_week | `index.dayofweek` | — | Thứ trong tuần (0-4) |
| **Calendar** | month | `index.month` | — | Tháng trong năm |

> **Tổng cộng: ~23 features** sau khi loại bỏ NaN từ rolling windows.

### 4.2 Target Variable

```
Label = 1 (Tăng)   nếu close[t+1] > close[t]
Label = 0 (Giảm)   nếu close[t+1] <= close[t]
```

> **Lưu ý quan trọng:** Target được shift(-1) trước khi train — tức là dùng features của ngày T để dự báo nhãn T+1. Hàng cuối cùng (ngày hôm nay) sẽ có features nhưng không có label — đây chính là input cho inference.

---

## 5. Model Training — `ml_models/trainer.py`

### 5.1 Train/Test Split Strategy

```
Dùng TimeSeriesSplit (k=5 folds) — KHÔNG dùng random shuffle
vì data là time-series có tính tuần tự.

Ví dụ với 504 ngày:
  Fold 1: Train[0:80]  → Test[80:100]
  Fold 2: Train[0:180] → Test[180:200]
  Fold 3: Train[0:280] → Test[280:300]
  Fold 4: Train[0:380] → Test[380:400]
  Fold 5: Train[0:480] → Test[480:504]  ← Final evaluation
```

### 5.2 XGBoost Hyperparameters

| Tham số | Giá trị | Lý do |
|---------|---------|--------|
| `n_estimators` | 500 | Đủ lớn với early stopping |
| `max_depth` | 4 | Tránh overfitting trên financial data |
| `learning_rate` | 0.05 | Conservative để generalize tốt hơn |
| `subsample` | 0.8 | Stochastic boosting |
| `colsample_bytree` | 0.8 | Feature subsampling |
| `scale_pos_weight` | auto | Tự tính từ class imbalance |
| `early_stopping_rounds` | 50 | Stop nếu không cải thiện 50 rounds |
| `eval_metric` | `logloss` | Phù hợp với binary classification |
| `random_state` | 42 | Reproducibility |

### 5.3 Evaluation Metrics

| Metric | Mục tiêu | Ngưỡng Chấp nhận |
|--------|----------|-----------------|
| Accuracy | Tỷ lệ đúng tổng thể | ≥ 55% |
| AUC-ROC | Khả năng phân biệt | ≥ 0.58 |
| Precision (UP) | Khi dự báo Tăng, đúng bao nhiêu? | ≥ 55% |
| Recall (UP) | Bắt được bao nhiêu ngày Tăng? | ≥ 50% |

> **Lưu ý:** Financial forecasting vốn có noise rất cao. Accuracy ~60% là acceptable nếu Precision cao — quan trọng hơn là signal chất lượng cao thay vì coverage.

### 5.4 Model Serialization

```
data/models/
├── xgboost_vnindex_v1_20260506.pkl     # joblib.dump(model)
├── feature_config_20260506.json        # {"features": [...], "version": "1"}
└── training_metadata_20260506.json     # {"accuracy": 0.58, "auc": 0.61, ...}
```

---

## 6. Inference — `ml_models/predictor.py`

### 6.1 Inference Flow

```
Input: Hàng cuối cùng của DataFrame sau Feature Engineering (ngày hôm nay)
        │
        ▼
Load model từ data/models/xgboost_vnindex_*.pkl
        │
        ▼
model.predict_proba(X_today)
        │
        ▼
Output: {
  "p_up": 0.72,
  "p_down": 0.28,
  "direction": "TĂNG",
  "confidence": "HIGH"   # HIGH nếu |p_up - 0.5| > 0.15
}
```

### 6.2 Confidence Thresholds

| P(Tăng) | Nhãn | Độ Tin Cậy |
|---------|------|------------|
| > 0.65 | TĂNG | HIGH |
| 0.55 – 0.65 | TĂNG | MEDIUM |
| 0.45 – 0.55 | ĐI NGANG | LOW |
| 0.35 – 0.45 | GIẢM | MEDIUM |
| < 0.35 | GIẢM | HIGH |

---

## 7. Dependency Map — Module 1

```
backend/
  scraper/market_data.py
      └── yfinance.download()
      └── pandas DataFrame cleanup

  ml_models/feature_engineer.py
      └── ta.momentum.*
      └── ta.trend.*
      └── ta.volatility.*
      └── ta.volume.*

  ml_models/trainer.py
      └── xgboost.XGBClassifier
      └── sklearn.model_selection.TimeSeriesSplit
      └── sklearn.metrics.*
      └── joblib (serialize)

  ml_models/predictor.py
      └── joblib.load()
      └── model.predict_proba()
```

---

## 8. API Endpoint Liên quan

- `POST /train` — Trigger training pipeline đầy đủ
- `POST /predict` — Gọi inference (sau đó pass vào LangGraph)
- `GET /market-data` — Trả về OHLCV + indicators để hiển thị trên UI
