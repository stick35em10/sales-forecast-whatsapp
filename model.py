import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os
from datetime import datetime, timedelta

class SalesForecaster:
    def __init__(self):
        self.model = None
        self.is_trained = False
        self.last_training_date = None
        
    def generate_sample_data(self, periods=365, store_name="Loja Principal"):
        """Gera dados de exemplo realistas para demonstração"""
        np.random.seed(42)
        
        dates = pd.date_range(start='2022-01-01', periods=periods, freq='D')
        
        # Tendência crescente
        trend = np.linspace(1000, 5000, periods)
        
        # Sazonalidade semanal
        day_of_week = dates.dayofweek
        seasonal_weekly = np.where(day_of_week >= 5, 800, -200)
        
        # Sazonalidade mensal
        day_of_month = dates.day
        seasonal_monthly = np.where(day_of_month >= 25, 500, 
                                   np.where(day_of_month <= 5, 300, 0))
        
        # Sazonalidade anual
        month = dates.month
        seasonal_yearly = np.where(month == 12, 1000,
                                  np.where(month == 11, 1500,
                                          np.where(month == 6, 600, 0)))
        
        # Eventos especiais
        special_events = np.zeros(periods)
        black_friday = (dates.month == 11) & (dates.day >= 25)
        christmas = (dates.month == 12) & (dates.day >= 15) & (dates.day <= 25)
        special_events[black_friday] = 2000
        special_events[christmas] = 1500
        
        # Ruído aleatório
        noise = np.random.normal(0, 300, periods)
        
        # Vendas totais
        sales = trend + seasonal_weekly + seasonal_monthly + seasonal_yearly + special_events + noise
        sales = np.maximum(sales, 100)
        
        # Criar DataFrame
        df = pd.DataFrame({
            'date': dates,
            'sales': sales.round(2),
            'day_of_week': day_of_week,
            'day_of_month': day_of_month,
            'month': month,
            'week_of_year': dates.isocalendar().week,
            'is_weekend': (day_of_week >= 5).astype(int),
            'is_month_end': (day_of_month >= 25).astype(int),
            'is_holiday_season': ((month == 11) | (month == 12)).astype(int),
            'store_name': store_name
        })
        
        return df
    
    def create_features(self, df):
        """Cria features para o modelo"""
        df = df.copy()
        
        # Features cíclicas
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)
        df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
        df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
        
        if 'sales' in df.columns:
            # Features de lag
            for lag in [1, 7, 30]:
                df[f'sales_lag_{lag}'] = df['sales'].shift(lag)
            
            # Médias móveis
            for window in [7, 14, 30]:
                df[f'sales_ma_{window}'] = df['sales'].rolling(window=window).mean()
        
        # Preencher valores NaN
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        return df
    
    def train_model(self, df=None, store_name="Loja Principal"):
        """Treina o modelo de previsão"""
        if df is None:
            df = self.generate_sample_data(store_name=store_name)
        
        print(f"🔄 Treinando modelo para {store_name}...")
        df = self.create_features(df)
        
        # Definir features
        feature_columns = [
            'day_sin', 'day_cos', 'month_sin', 'month_cos',
            'day_of_month', 'week_of_year', 'is_weekend', 
            'is_month_end', 'is_holiday_season',
            'sales_lag_1', 'sales_lag_7', 'sales_lag_30',
            'sales_ma_7', 'sales_ma_14', 'sales_ma_30'
        ]
        
        # Remover linhas com NaN
        df_clean = df.dropna()
        
        X = df_clean[feature_columns]
        y = df_clean['sales']
        
        # Split dos dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Treinar modelo
        self.model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
        
        self.model.fit(X_train, y_train)
        
        # Avaliar modelo
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        self.is_trained = True
        self.last_training_date = datetime.now()
        
        # Salvar modelo
        model_filename = f'sales_model_{store_name.replace(" ", "_").lower()}.joblib'
        joblib.dump(self.model, model_filename)
        
        print(f"✅ Modelo treinado para {store_name}! MAE: R$ {mae:.2f}, RMSE: R$ {rmse:.2f}")
        
        return {'mae': mae, 'rmse': rmse, 'store_name': store_name}
    
    def forecast(self, days=30, store_name="Loja Principal"):
        """Faz previsão para os próximos dias"""
        if not self.is_trained:
            model_filename = f'sales_model_{store_name.replace(" ", "_").lower()}.joblib'
            if os.path.exists(model_filename):
                self.model = joblib.load(model_filename)
                self.is_trained = True
                print(f"✅ Modelo carregado para {store_name}")
            else:
                print(f"🔄 Treinando novo modelo para {store_name}...")
                self.train_model(store_name=store_name)
        
        # Gerar dados históricos
        historical_data = self.generate_sample_data(store_name=store_name)
        historical_data = self.create_features(historical_data)
        
        # Criar datas futuras
        last_date = historical_data['date'].max()
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=days,
            freq='D'
        )
        
        # Preparar dados futuros
        future_data = []
        last_known = historical_data.iloc[-1].to_dict()
        
        for i, date in enumerate(future_dates):
            row = {
                'date': date,
                'day_of_week': date.dayofweek,
                'day_of_month': date.day,
                'month': date.month,
                'week_of_year': date.isocalendar().week,
                'is_weekend': 1 if date.dayofweek >= 5 else 0,
                'is_month_end': 1 if date.day >= 25 else 0,
                'is_holiday_season': 1 if date.month in [11, 12] else 0,
                'sales_lag_1': last_known['sales'],
                'sales_lag_7': last_known.get('sales_lag_1', last_known['sales']),
                'sales_lag_30': last_known.get('sales_lag_7', last_known['sales']),
                'sales_ma_7': last_known.get('sales_ma_7', last_known['sales']),
                'sales_ma_14': last_known.get('sales_ma_14', last_known['sales']),
                'sales_ma_30': last_known.get('sales_ma_30', last_known['sales'])
            }
            future_data.append(row)
        
        future_df = pd.DataFrame(future_data)
        future_df = self.create_features(future_df)
        
        # Features para previsão
        feature_columns = [
            'day_sin', 'day_cos', 'month_sin', 'month_cos',
            'day_of_month', 'week_of_year', 'is_weekend', 
            'is_month_end', 'is_holiday_season',
            'sales_lag_1', 'sales_lag_7', 'sales_lag_30',
            'sales_ma_7', 'sales_ma_14', 'sales_ma_30'
        ]
        
        # Garantir que todas as features existam
        for col in feature_columns:
            if col not in future_df.columns:
                future_df[col] = 0
        
        # Fazer previsões
        predictions = self.model.predict(future_df[feature_columns])
        
        result_df = pd.DataFrame({
            'date': future_dates,
            'predicted_sales': predictions,
            'day_name': future_dates.day_name(),
            'is_weekend': future_df['is_weekend'],
            'is_month_end': future_df['is_month_end']
        })
        
        print(f"✅ Previsão gerada para {days} dias - {store_name}")
        return result_df

    def get_forecast_stats(self, forecast_df, store_name="Loja Principal"):
        """Calcula estatísticas da previsão"""
        total_sales = forecast_df['predicted_sales'].sum()
        avg_daily = forecast_df['predicted_sales'].mean()
        max_day = forecast_df.loc[forecast_df['predicted_sales'].idxmax()]
        min_day = forecast_df.loc[forecast_df['predicted_sales'].idxmin()]
        
        # Previsão por dia da semana
        weekday_stats = forecast_df.groupby('day_name')['predicted_sales'].mean()
        
        stats = {
            'store_name': store_name,
            'forecast_days': len(forecast_df),
            'total_predicted': total_sales,
            'daily_average': avg_daily,
            'best_day': {
                'date': max_day['date'].strftime('%d/%m/%Y'),
                'day_name': max_day['day_name'],
                'sales': max_day['predicted_sales']
            },
            'worst_day': {
                'date': min_day['date'].strftime('%d/%m/%Y'),
                'day_name': min_day['day_name'],
                'sales': min_day['predicted_sales']
            },
            'weekday_averages': weekday_stats.to_dict(),
            'generated_at': datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        }
        
        return stats

# Teste do modelo
if __name__ == "__main__":
    print("🧪 Testando o modelo...")
    forecaster = SalesForecaster()
    
    metrics = forecaster.train_model(store_name="Loja Centro")
    forecast = forecaster.forecast(days=30, store_name="Loja Centro")
    stats = forecaster.get_forecast_stats(forecast, "Loja Centro")
    
    print(f"\n📊 ESTATÍSTICAS:")
    print(f"Loja: {stats['store_name']}")
    print(f"Total Previsto: R$ {stats['total_predicted']:,.2f}")
    print(f"Média Diária: R$ {stats['daily_average']:,.2f}")
    print(f"Melhor Dia: {stats['best_day']['date']} - R$ {stats['best_day']['sales']:,.2f}")
    
    print(f"\n📅 PRIMEIRAS PREVISÕES:")
    print(forecast.head(10).to_string(index=False))
