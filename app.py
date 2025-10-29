from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
import json
from datetime import datetime
from model import SalesForecaster
import logging
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse
import traceback

# Configuração
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_WHATSAPP_NUMBER = os.getenv('TWILIO_WHATSAPP_NUMBER')

app = Flask(__name__)
CORS(app)

# Inicializar cliente Twilio
twilio_client = None
if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
    try:
        twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        print("✅ Twilio client inicializado")
    except Exception as e:
        print(f"⚠️ Twilio não configurado: {e}")
else:
    print("⚠️ Variáveis Twilio não configuradas")

# Inicializar forecasters
forecasters = {
    "centro": SalesForecaster(),
    "shopping": SalesForecaster(),
    "online": SalesForecaster()
}

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def send_whatsapp_message(to, message):
    """Envia mensagem via WhatsApp usando Twilio"""
    if not twilio_client:
        logger.warning("Twilio não configurado - mensagem simulada")
        print(f"📱 WHATSAPP SIMULADO para {to}: {message[:100]}...")
        return True
    
    try:
        message = twilio_client.messages.create(
            from_=f'whatsapp:{TWILIO_WHATSAPP_NUMBER}',
            to=f'whatsapp:{to}',
            body=message
        )
        logger.info(f"Mensagem enviada para {to} - SID: {message.sid}")
        return True
    except Exception as e:
        logger.error(f"Erro ao enviar WhatsApp: {str(e)}")
        return False

def format_currency(value):
    """Formata valor como moeda brasileira"""
    return f"R$ {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def create_sales_plot(forecast_df, store_name):
    """Cria gráfico de previsão de vendas"""
    try:
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('📈 Previsão de Vendas - 30 Dias', '📊 Vendas por Dia da Semana'),
            vertical_spacing=0.1
        )
        
        # Gráfico de linha principal
        fig.add_trace(
            go.Scatter(
                x=forecast_df['date'],
                y=forecast_df['predicted_sales'],
                mode='lines+markers',
                name='Vendas Previstas',
                line=dict(color='#2E86AB', width=3),
                marker=dict(size=6)
            ),
            row=1, col=1
        )
        
        # Gráfico de barras por dia da semana
        weekday_sales = forecast_df.groupby('day_name')['predicted_sales'].mean()
        
        fig.add_trace(
            go.Bar(
                x=weekday_sales.index,
                y=weekday_sales.values,
                name='Média por Dia',
                marker_color='#A23B72'
            ),
            row=2, col=1
        )
        
        # Atualizar layout
        fig.update_layout(
            height=600,
            showlegend=False,
            title_text=f"🏪 Previsão de Vendas - {store_name}",
            title_x=0.5
        )
        
        # Converter para imagem
        img_bytes = fig.to_image(format="png", width=800, height=600, scale=1)
        return img_bytes
    except Exception as e:
        logger.error(f"Erro ao criar gráfico: {str(e)}")
        return None

def generate_forecast_report(store_name, days=30):
    """Gera relatório completo de previsão"""
    store_key = store_name.lower().replace(" ", "")
    if store_key not in forecasters:
        store_key = "centro"
    
    forecaster = forecasters[store_key]
    
    # Gerar previsão
    forecast_df = forecaster.forecast(days=days, store_name=store_name)
    stats = forecaster.get_forecast_stats(forecast_df, store_name)
    
    # Criar gráfico
    plot_bytes = create_sales_plot(forecast_df, store_name)
    
    # Gerar texto formatado para WhatsApp
    report_text = f"""
🏪 *RELATÓRIO DE PREVISÃO - {store_name.upper()}*

📅 Período: {days} dias
🕒 Gerado em: {stats['generated_at']}

💰 *RESUMO FINANCEIRO:*
• Total Previsto: {format_currency(stats['total_predicted'])}
• Média Diária: {format_currency(stats['daily_average'])}

🎯 *MELHORES MOMENTOS:*
🏆 Melhor Dia: {stats['best_day']['date']} ({stats['best_day']['day_name']})
💵 Vendas: {format_currency(stats['best_day']['sales'])}

📉 *DIAS ATENÇÃO:*
⚠️  Pior Dia: {stats['worst_day']['date']} ({stats['worst_day']['day_name']})
💵 Vendas: {format_currency(stats['worst_day']['sales'])}

📊 *PRÓXIMOS 5 DIAS:*
"""
    
    # Adicionar próximos 5 dias
    for _, row in forecast_df.head().iterrows():
        report_text += f"• {row['date'].strftime('%d/%m')} ({row['day_name']}): {format_currency(row['predicted_sales'])}\n"
    
    report_text += """
💡 *RECOMENDAÇÕES:*
• Prepare estoque para os melhores dias
• Otimize equipe nos fins de semana
• Promoções nos dias mais fracos

_Relatório automático - Sistema de Previsão de Vendas_
"""
    
    return {
        'text': report_text,
        'plot_bytes': plot_bytes,
        'stats': stats,
        'forecast_data': forecast_df.to_dict('records')
    }

@app.route('/')
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>🤖 Previsão de Vendas - WhatsApp</title>
        <meta charset="utf-8">
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .status { padding: 20px; background: #f0f8ff; border-radius: 10px; }
            .api-endpoint { background: #2c3e50; color: white; padding: 10px; border-radius: 5px; margin: 5px 0; font-family: monospace; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🏪 Sistema de Previsão de Vendas</h1>
            <h2>🤖 Integração WhatsApp + Machine Learning</h2>
            
            <div class="status">
                <h3>✅ Sistema Operacional</h3>
                <p><strong>Status:</strong> Online</p>
                <p><strong>Modelo:</strong> Machine Learning (Random Forest)</p>
                <p><strong>WhatsApp:</strong> Twilio API</p>
            </div>

            <h3>📱 Como usar:</h3>
            <p>Envie "PREVISAO" para o número WhatsApp do bot</p>
            <p>Receba relatório completo com previsões de 30 dias</p>

            <h3>🔧 Endpoints API:</h3>
            <div class="api-endpoint">GET / - Esta página</div>
            <div class="api-endpoint">POST /webhook - Webhook WhatsApp</div>
            <div class="api-endpoint">GET /forecast?store=Loja-Centro - Previsão JSON</div>
            <div class="api-endpoint">GET /health - Health Check</div>
        </div>
    </body>
    </html>
    """

@app.route('/webhook', methods=['POST'])
def whatsapp_webhook():
    """Webhook para receber mensagens do WhatsApp"""
    try:
        incoming_msg = request.values.get('Body', '').strip().upper()
        from_number = request.values.get('From', '').replace('whatsapp:', '')
        
        logger.info(f"📱 Mensagem de {from_number}: {incoming_msg}")
        
        response = MessagingResponse()
        
        if incoming_msg in ['PREVISAO', 'PREVISÃO', 'RELATORIO']:
            # Mensagem de processamento
            response.message("🔄 Gerando seu relatório de previsão... Aguarde!")
            
            # Em produção real, usar queue/celery
            import threading
            thread = threading.Thread(
                target=send_forecast_report, 
                args=(from_number, "Loja Centro")
            )
            thread.start()
            
        elif incoming_msg in ['AJUDA', 'HELP']:
            help_text = """
🤖 *COMANDOS DISPONÍVEIS:*

*PREVISAO* - 📊 Gerar relatório de previsão
*AJUDA* - ❓ Mostrar ajuda

_Envie PREVISAO para começar!_
"""
            response.message(help_text)
            
        else:
            welcome_text = """
🏪 *Bem-vindo ao Sistema de Previsão!*

Envie *PREVISAO* para receber um relatório completo de previsão de vendas para os próximos 30 dias.

Inclui:
• Previsões diárias
• Estatísticas detalhadas  
• Gráficos visuais
• Recomendações
"""
            response.message(welcome_text)
        
        return str(response)
        
    except Exception as e:
        logger.error(f"Erro no webhook: {str(e)}")
        response = MessagingResponse()
        response.message("❌ Erro ao processar. Tente novamente.")
        return str(response)

def send_forecast_report(phone_number, store_name):
    """Envia relatório via WhatsApp"""
    try:
        logger.info(f"📊 Enviando relatório para {phone_number}")
        
        report = generate_forecast_report(store_name)
        success = send_whatsapp_message(phone_number, report['text'])
        
        if success:
            logger.info(f"✅ Relatório enviado para {phone_number}")
        else:
            logger.error(f"❌ Falha ao enviar para {phone_number}")
            
    except Exception as e:
        logger.error(f"Erro ao enviar relatório: {str(e)}")

@app.route('/forecast', methods=['GET'])
def get_forecast():
    """Endpoint para obter previsão em JSON"""
    try:
        store_name = request.args.get('store', 'Loja Centro')
        days = int(request.args.get('days', '30'))
        
        report = generate_forecast_report(store_name, days)
        
        response_data = {
            'success': True,
            'store': store_name,
            'days': days,
            'stats': report['stats'],
            'forecast': report['forecast_data'],
            'generated_at': datetime.now().isoformat()
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Erro na previsão: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'sales-forecast-whatsapp',
        'timestamp': datetime.now().isoformat(),
        'twilio_configured': twilio_client is not None
    })

# Inicializar modelos
@app.before_first_request
def initialize_models():
    """Inicializa os modelos na startup"""
    try:
        logger.info("🚀 Inicializando modelos...")
        for store_name, forecaster in forecasters.items():
            forecaster.train_model(store_name=f"Loja {store_name.title()}")
        logger.info("✅ Modelos inicializados!")
    except Exception as e:
        logger.error(f"❌ Erro na inicialização: {str(e)}")

if __name__ == '__main__':
    # Inicializar modelos
    initialize_models()
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
