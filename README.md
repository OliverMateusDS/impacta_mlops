# Projeto MLOps – Previsão de Preço de Diamantes

Este repositório faz parte da disciplina **MLOps – Running ML in Production Environments**.

O objetivo do projeto é mostrar, passo a passo, como evoluir de um fluxo manual em notebooks para um projeto de Machine Learning organizado, reprodutível e pronto para automação, seguindo boas práticas de MLOps.

---

## Contexto até aqui

Na **Aula 1**, trabalhamos com um fluxo típico de ciência de dados:

- EDA em notebook
- treino e inferência manual
- uso inicial do MLflow para registrar experimentos

Esse fluxo funciona, mas não escala e não é fácil de repetir.

Na **Aula 2**, o foco foi **organizar o projeto e estruturar o ciclo de dados**, preparando o terreno para automação e evolução do pipeline.

Na **Aula 3**, o projeto evoluiu para um **pipeline completo de modelagem**, com rastreabilidade e testes automatizados.

Na **Aula 4**, fechamos o ciclo com **deploy e operação**, colocando **MLflow + aplicação** para rodar via **Docker Compose**, além de **CI** com GitHub Actions.

---

## O que foi feito na Aula 3

Foram implementados:

- separação da lógica de modelagem em módulos Python
- pipeline de pré-processamento e treino com scikit-learn
- script de treino executável via linha de comando (`train.py`)
- avaliação padronizada de métricas de regressão
- experiment tracking com MLflow
- testes automatizados com pytest

A partir deste ponto, o modelo deixa de depender do notebook e passa a ser tratado como um **artefato rastreável**.

---

## Aula 4 – Deploy, Operação e Ciclo Completo de MLOps

### Arquitetura adotada

Para simplificar a execução e garantir reprodutibilidade, foi adotado:

- **MLflow Server em Docker**
  - tracking e histórico de experimentos
  - persistência local via volume (`./mlruns`)

- **Aplicação Streamlit em Docker**
  - UI para inferência em tempo real
  - consome o modelo treinado (modelo salvo localmente)
  - preparada para evoluir para consumo via Model Registry

### CI (GitHub Actions)

- workflow de CI configurado para rodar `pytest` a cada `push` e `pull_request`.

---

## Estrutura atual do projeto

```text
impacta_mlops/
│
├── notebooks/
│   └── eda_diamonds.ipynb
│
├── src/
│   ├── data.py
│   ├── model.py
│   ├── evaluate.py
│   └── __init__.py
│
├── app/
│   └── streamlit_app.py
│
├── tests/
│   ├── test_data.py
│   ├── test_model.py
│   ├── test_train.py
│   └── __init__.py
│
├── requirements.txt
├── train.py
├── docker-compose.yml
├── Dockerfile
├── pytest.ini
├── README.md
└── .gitignore


