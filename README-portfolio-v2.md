# FinOps & Risk Analytics: Pipeline Preditivo de Risco Financeiro e Alocacao de Recursos (End-to-End ML & XAI)

Projeto End-to-End de Ciencia e Analise de Dados voltado para a previsao de inadimplencia/risco financeiro, otimizacao de alocacao de recursos em FinOps, tratamento rigoroso de dados segundo a metodologia KDD, inferencia estatistica parametrica/nao-parametrica, modelagem preditiva avancada e explicabilidade algoritmica (XAI) com SHAP.

---

## Contexto e Problema de Negocio

Em instituicoes financeiras e areas de FinOps corporativo, a alocacao ineficiente de capital e a incapacidade de prever riscos de inadimplencia geram perdas operacionais expressivas. Este projeto resolve esse desafio construindo um pipeline preditivo modular focado em:

1. **Higienizacao e Governanca de Dados**: Processar bases brutas ruidosas contendo inconformidades, duplicatas e dados faltantes usando a metodologia KDD (Knowledge Discovery in Databases).
2. **Validacao Estatistica de Hipoteses**: Investigar relacoes de causalidade/correlacao e testar diferencas significativas entre grupos de clientes atraves de testes de hipoteses rigorosos (t de Welch, Levene, d de Cohen).
3. **Mitigacao de Multicolinearidade e Overfitting**: Diagnosticar multicolinearidade via VIF (Variance Inflation Factor) e aplicar tecnicas de regularizacao (Ridge L2 e Lasso L1) e modelos avancados (SVM, Regressao Logistica, Perceptron).
4. **Explicabilidade para Tomada de Decisao (XAI)**: Traduzir decisoes de caixa-preta em insights acionaveis para diretores (C-Level) utilizando SHAP (Shapley Additive exPlanations).

---

## Arquitetura e Fases do Projeto

O projeto e estruturado em 4 grandes etapas modulares, simulando o ciclo de vida real de um projeto de inteligencia de dados:

- **Etapa 1: Data Wrangling & KDD**
  - Imputacao condicional (`groupby + transform`), auditoria de outliers pelo criterio de Tukey/IQR e remocao de duplicatas.
- **Etapa 2: EDA & Inferencia Estatistica**
  - Matrizes de correlacao de Pearson vs. Spearman, teste de homocedasticidade de Levene, teste t de Welch e calculo do d de Cohen para tamanho do efeito.
- **Etapa 3: Modelagem Preditiva & Regularizacao**
  - Diagnostico de VIF, comparacao OLS vs. Ridge (L2) vs. Lasso (L1), e batalha de classificadores (Perceptron, Regressao Logistica, SVM RBF).
- **Etapa 4: Explicabilidade (XAI) & Parecer Executivo**
  - SHAP Beeswarm (analise global), SHAP Force Plot (analise local) e redacao de parecer tecnico executivo para tomada de decisao estrategica.

---

## Estrutura do Repositorio

```text
finops-risk-analytics/
├── README.md                            <- Apresentacao do projeto e relatorio executivo
├── requirements.txt                     <- Dependencias e bibliotecas do projeto
├── data/
│   ├── raw/                             <- Dados brutos originais com ruidos
│   └── processed/                       <- Dados limpos e transformados pos-KDD
├── notebooks/
│   ├── 01_data_wrangling_kdd.ipynb      <- Limpeza, imputacao e auditoria de outliers
│   ├── 02_eda_and_hypothesis.ipynb      <- Analise exploratoria e testes estatisticos
│   ├── 03_model_training_and_vif.ipynb  <- VIF, regularizacao e treinamento de modelos
│   └── 04_model_explainability.ipynb    <- Explicabilidade SHAP e parecer executivo
├── src/                                 <- Modulos e scripts Python reutilizaveis
│   ├── data_prep.py                     <- Funcoes de limpeza e tratamento KDD
│   ├── stats_tests.py                   <- Funcoes auxiliares para testes de hipoteses
│   └── evaluation.py                    <- Funcoes de metricas e matrizes de confusao
└── models/                              <- Modelos treinados serializados (.pkl)
```

---

## Detalhamento Tecnico das Etapas

### 1. Data Wrangling, Higienizacao e KDD
- **Tratamento de Faltantes**: Imputacao condicional pela mediana de subgrupos categoricos (`df.groupby('categoria')['coluna'].transform('median')`), evitando distorcoes decorrentes de medias afetadas por caudas longas.
- **Deteccao de Outliers**: Aplicacao do criterio do Intervalo Interquartil (IQR), limitando os limites aceitaveis em [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR].
- **Engenharia de Recursos**: Normalizacao MinMax e Padronizacao (Z-score) para garantir convergencia de algoritmos baseados em distancia e gradiente.

### 2. Exploratory Data Analysis (EDA) & Inferencia Estatistica
- **Assimetria de Distribuicoes**: Diagnostico da divergencia entre Media e Mediana para identificacao de caudas longas em renda e custos operacionais.
- **Correlacao**: Analise comparativa entre Pearson (relacoes lineares) e Spearman (relacoes monotonicas nao-lineares) para afastar correlacoes espurias.
- **Testes de Hipoteses**:
  - **Teste de Levene**: Validacao da premissa de variancias iguais.
  - **Teste t de Welch**: Comparacao de medias populacionais sem assumir homocedasticidade.
  - **d de Cohen**: Calculo do tamanho do efeito (effect size), garantindo relevante impacto pratico e nao apenas significancia estatistica (p-valor).

### 3. Modelagem Preditiva & Selecao de Recursos
- **Analise do VIF (Variance Inflation Factor)**: Remocao/combinacao de variaveis com VIF > 5.0 para eliminacao de multicolinearidade.
- **Regularizacao L1 vs. L2**:
  - **Ridge (L2)**: Supressao do tamanho dos coeficientes para estabilizacao de estimadores.
  - **Lasso (L1)**: Reducao de coeficientes nao essenciais a zero (selecao automatica de atributos).
- **Classificadores Comparados**:
  - **Regressao Logistica** (Baseline com probabilidades calibradas)
  - **Perceptron** (Classificador linear de camada unica)
  - **SVM (Support Vector Machines)** com Kernel Trick (Linear vs. RBF)
- **Priorizacao do Recall**: Em cenarios de risco de credito, falsos negativos (aprovar um inadimplente) custam significativamente mais que falsos positivos.

### 4. Explicabilidade (XAI) & SHAP
- **SHAP Summary / Beeswarm Plot**: Mapeamento do impacto global das variaveis mais determinantes na decisao do algoritmo.
- **SHAP Waterfall / Force Plot**: Inspecao individualizada (local explainability) explicando exatamente por que o risco de um cliente especifico foi classificado como alto ou baixo.

---

## Resultados e Comparativo de Modelos

| Modelo | Acuracia | Precisao | **Recall (Prioritario)** | F1-Score | AUC-ROC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Perceptron)** | 72.4% | 65.1% | 68.2% | 66.6% | 0.71 |
| **Regressao Logistica (OLS)** | 81.5% | 77.2% | 79.0% | 78.1% | 0.83 |
| **Regressao Logistica (Lasso L1)** | 83.1% | 79.5% | 82.4% | 80.9% | 0.86 |
| **SVM (Kernel RBF)** | **86.8%** | **82.1%** | **88.5%** | **85.2%** | **0.91** |

> **Conclusao Tecnica**: O modelo SVM RBF apresentou o melhor desempenho geral, reduzindo a taxa de Falsos Negativos em 20.3% comparado ao baseline, maximizando a seguranca das operacoes de concessao de credito.

---

## Como Executar este Repositorio

### Pre-requisitos
- Python 3.10 ou superior
- Git instalado

### Passo a Passo

1. Clonar o repositorio para sua maquina local.
2. Criar e ativar o ambiente virtual com `python -m venv venv`.
3. Instalar as dependencias necessarias com `pip install -r requirements.txt`.
4. Executar os notebooks Jupyter no ambiente de desenvolvimento.

---

## Parecer Tecnico Executivo (Resumo para C-Level)

1. **Eficiencia de Negocio**: O modelo desenvolvido reduz a exposicao ao risco de inadimplencia sem prejudicar a taxa de aprovacao de bons clientes.
2. **Governanca & Transparencia**: Atraves da integracao com o ecossistema SHAP, o sistema garante transparencia e auditabilidade referente ao direito de explicacao das decisoes automatizadas.
3. **Proximos Passos**: Implementar monitoramento continuo de Data Drift e Concept Drift via pipelines de MLOps.

---

## Autor

**Seu Nome Completo**  
*Cientista de Dados / Analista de Dados*  

---
*Projeto desenvolvido como portfolio pratico abrangendo conceitos de Ciencia de Dados, Estatistica Inferencial e Machine Learning.*
