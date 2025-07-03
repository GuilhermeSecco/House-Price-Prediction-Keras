# 🏠 Predição de Preços de Imóveis com Redes Neurais (Keras)

## 📌 Descrição

Este projeto aplica **Deep Learning** para prever o preço de imóveis com base em características como número de quartos, banheiros, área construída, localização e condição da propriedade. A base de dados utilizada é o famoso dataset `kc_house_data.csv`, que contém dados de casas vendidas na região de King County, nos EUA.

O objetivo principal é desenvolver um modelo de regressão com **Keras** (usando `Sequential`) capaz de realizar predições precisas de preço com base em atributos numéricos.

---

## 🔧 Tecnologias utilizadas

- Python
- Pandas & NumPy
- Scikit-learn (MinMaxScaler, train_test_split)
- Keras / TensorFlow (Sequential, Dense)
- Matplotlib

---

## 📊 Etapas do Projeto

- Importação e limpeza dos dados
- Seleção das variáveis relevantes
- Normalização com MinMaxScaler
- Divisão entre treino e teste
- Construção da rede neural com Keras
- Treinamento do modelo
- Avaliação com métricas de regressão
- Visualizações gráficas:
   - Real vs Previsto
   - Evolução da loss por epoch
     
---

## 📈 Resultados

- Erro percentual médio: Aproximadamente X% (valor depende da execução)
- Acurácia visual: Mostrada por meio de gráfico Real vs Previsto
- Convergência: Acompanhada pela curva de perda (loss/val_loss)

---

## 🧠 Conclusão

Este projeto demonstra como redes neurais podem ser utilizadas para tarefas de regressão não linear, aplicadas a problemas reais como avaliação de preços imobiliários. Com o uso de normalização, arquitetura simples e boas práticas de separação de dados, foi possível obter bons resultados preditivos.

---
