# Sistema preditivo de obesidade

Projeto de Machine Learning para prever o nível de obesidade a partir de variáveis clínicas, comportamentais e demográficas.

## Estrutura

```text
app/streamlit_app.py
src/features.py
src/train.py
data/Obesity.csv
models/obesity_model.joblib
reports/metrics.json
reports/classification_report.csv
reports/confusion_matrix.csv
entrega.txt
requirements.txt
```

## Execução local

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src/train.py
streamlit run app/streamlit_app.py
```

## Deploy no Streamlit Community Cloud

1. Subir o repositório no GitHub.
2. Acessar o Streamlit Community Cloud.
3. Criar um app apontando para `app/streamlit_app.py`.
4. Confirmar que `requirements.txt`, `data/`, `models/` e `reports/` estão versionados.
