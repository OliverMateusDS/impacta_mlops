# Diamonds Price Prediction – MLOps Project

Projeto de Machine Learning com foco em boas práticas de **MLOps**: pipeline reprodutível de treino, rastreabilidade de experimentos com **MLflow**, aplicação de inferência em **Streamlit**, containerização com **Docker Compose** e **CI** (GitHub Actions) executando testes automatizados.

---

## Objetivo

Prever o **preço de diamantes** com base em características do dataset `diamonds` (seaborn), aplicando um fluxo completo de MLOps:

- pipeline de modelagem (pré-processamento + treino)
- tracking de métricas/params/artefatos no MLflow
- inferência via aplicação Streamlit
- execução reprodutível com Docker
- testes automatizados e CI

---

## Arquitetura

- **MLflow Server (Docker)**
  - tracking e histórico de experimentos
  - persistência local via volume `./mlruns`

- **Streamlit App (Docker)**
  - interface de inferência em tempo real
  - consome o modelo treinado (salvo localmente)
  - preparado para evoluir para consumo via MLflow Registry

---

## Requisitos

- **Docker Desktop** (com WSL2 no Windows)
- (Opcional para rodar sem Docker) **Python 3.12+**

---

## Como executar com Docker (recomendado)

Na raiz do projeto:

```bash
docker compose up --build
