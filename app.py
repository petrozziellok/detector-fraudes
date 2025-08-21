# app.py
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar modelo
model = joblib.load("modelo_rf.pkl")

st.title("💳 Detecção de Fraudes em Cartões de Crédito")
st.write("Faça upload de um CSV com as mesmas colunas usadas para treinar o modelo.")


expected_cols = [
    'Time',
    'V1','V2','V3','V4','V5','V6','V7','V8','V9','V10',
    'V11','V12','V13','V14','V15','V16','V17','V18','V19','V20',
    'V21','V22','V23','V24','V25','V26','V27','V28','Amount'
]

uploaded_file = st.file_uploader("Faça upload de um CSV com transações", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    
   
    if 'Class' in data.columns:
        data = data.drop('Class', axis=1)
    
    
    if 'Time' not in data.columns:
        data['Time'] = 0
    
    missing_cols = set(expected_cols) - set(data.columns)
    if missing_cols:
        st.error(f"Faltam colunas no CSV: {missing_cols}")
    else:
        
        data = data[expected_cols]
        
        # Fazer previsão
        predictions = model.predict(data)
        probabilities = model.predict_proba(data)[:,1]
        
        data['Fraude?'] = predictions
        data['Probabilidade'] = probabilities
        
        total_transacoes = len(data)
        total_fraudes = data['Fraude?'].sum()
        avg_prob = data['Probabilidade'].mean()
        
        st.subheader("📊 Resumo de Insights")
        st.write(f"- Total de transações analisadas: **{total_transacoes}**")
        st.write(f"- Total de fraudes detectadas: **{total_fraudes}**")
        st.write(f"- Probabilidade média de fraude: **{avg_prob:.2f}**")
        st.write(f"- Percentual de fraudes: **{(total_fraudes/total_transacoes*100):.2f}%**")
        
        st.subheader("📈 Distribuição das Probabilidades de Fraude")
        fig, ax = plt.subplots(figsize=(10,4))
        sns.histplot(data['Probabilidade'], bins=50, kde=True, color='skyblue', ax=ax)
        ax.set_xlabel("Probabilidade de Fraude")
        ax.set_ylabel("Número de Transações")
        st.pyplot(fig)
        
        st.subheader("🚨 Transações com maior risco")
        high_risk = data[data['Fraude?']==1].sort_values(by='Probabilidade', ascending=False)
        if not high_risk.empty:
            st.dataframe(high_risk[['Time','Amount','Fraude?','Probabilidade']])
        else:
            st.write("Nenhuma fraude detectada.")
        
        st.write("💡 Dica: as transações destacadas são consideradas fraudes pelo modelo. Use a coluna 'Probabilidade' para priorizar análise.")
