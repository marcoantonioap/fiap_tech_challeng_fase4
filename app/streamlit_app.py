from pathlib import Path
import sys
import json
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
from features import ObesityFeatureEngineer

DATA_PATH = ROOT / "data" / "Obesity.csv"
MODEL_PATH = ROOT / "models" / "obesity_model.joblib"
METRICS_PATH = ROOT / "reports" / "metrics.json"

st.set_page_config(page_title="Predição de Obesidade", page_icon="🏥", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["BMI"] = df["Weight"] / (df["Height"] ** 2)
    df["FCVC_cat"] = df["FCVC"].round().clip(1, 3).map({1: "Raramente", 2: "Às vezes", 3: "Sempre"})
    df["CH2O_cat"] = df["CH2O"].round().clip(1, 3).map({1: "< 1 L/dia", 2: "1-2 L/dia", 3: "> 2 L/dia"})
    df["FAF_cat"] = df["FAF"].round().clip(0, 3).map({0: "Nenhuma", 1: "1-2x/sem", 2: "3-4x/sem", 3: "5x+/sem"})
    return df

@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        from train import train
        train()
        return joblib.load(MODEL_PATH)

@st.cache_data
def load_metrics():
    with METRICS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)

def input_dataframe(values):
    return pd.DataFrame([values])

model = load_model()
df = load_data()
metrics = load_metrics()

st.title("Sistema preditivo para apoio à avaliação de obesidade")
st.caption("Ferramenta de apoio analítico para triagem. O resultado não substitui avaliação clínica.")

page = st.sidebar.radio("Navegação", ["Predição", "Painel analítico", "Desempenho do modelo"])

if page == "Predição":
    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox("Gênero", ["Female", "Male"])
        age = st.number_input("Idade", min_value=14.0, max_value=100.0, value=30.0, step=1.0)
        height = st.number_input("Altura em metros", min_value=1.20, max_value=2.20, value=1.70, step=0.01)
        weight = st.number_input("Peso em kg", min_value=30.0, max_value=250.0, value=80.0, step=0.5)
        family_history = st.selectbox("Histórico familiar de excesso de peso", ["yes", "no"])
    with col2:
        favc = st.selectbox("Consumo frequente de alimentos calóricos", ["yes", "no"])
        fcvc = st.slider("Frequência de vegetais", 1.0, 3.0, 2.0, 1.0)
        ncp = st.slider("Refeições principais por dia", 1.0, 4.0, 3.0, 1.0)
        caec = st.selectbox("Consumo entre refeições", ["no", "Sometimes", "Frequently", "Always"])
        smoke = st.selectbox("Fuma", ["no", "yes"])
    with col3:
        ch2o = st.slider("Consumo diário de água", 1.0, 3.0, 2.0, 1.0)
        scc = st.selectbox("Monitora calorias", ["no", "yes"])
        faf = st.slider("Atividade física semanal", 0.0, 3.0, 1.0, 1.0)
        tue = st.slider("Uso de dispositivos eletrônicos", 0.0, 2.0, 1.0, 1.0)
        calc = st.selectbox("Consumo de álcool", ["no", "Sometimes", "Frequently", "Always"])
        mtrans = st.selectbox("Meio de transporte", ["Automobile", "Motorbike", "Bike", "Public_Transportation", "Walking"])
    values = {
        "Gender": gender,
        "Age": age,
        "Height": height,
        "Weight": weight,
        "family_history": family_history,
        "FAVC": favc,
        "FCVC": fcvc,
        "NCP": ncp,
        "CAEC": caec,
        "SMOKE": smoke,
        "CH2O": ch2o,
        "SCC": scc,
        "FAF": faf,
        "TUE": tue,
        "CALC": calc,
        "MTRANS": mtrans
    }
    patient = input_dataframe(values)
    prediction = model.predict(patient)[0]
    probabilities = pd.DataFrame({"Classe": model.classes_, "Probabilidade": model.predict_proba(patient)[0]}).sort_values("Probabilidade", ascending=False)
    bmi = weight / (height ** 2)
    st.subheader("Resultado")
    c1, c2, c3 = st.columns(3)
    c1.metric("Classe prevista", prediction.replace("_", " "))
    c2.metric("IMC calculado", f"{bmi:.1f}")
    c3.metric("Confiança", f"{probabilities.iloc[0]['Probabilidade']:.1%}")
    st.plotly_chart(px.bar(probabilities, x="Classe", y="Probabilidade", text_auto=".1%", title="Distribuição de probabilidade por classe"), use_container_width=True)

if page == "Painel analítico":
    st.subheader("Visão analítica da base")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Registros", f"{len(df):,}".replace(",", "."))
    k2.metric("Idade média", f"{df['Age'].mean():.1f}")
    k3.metric("IMC médio", f"{df['BMI'].mean():.1f}")
    k4.metric("Classes", df["Obesity"].nunique())
    col1, col2 = st.columns(2)
    with col1:
        obesity_count = df["Obesity"].value_counts().reset_index()
        obesity_count.columns = ["Classe", "Total"]
        st.plotly_chart(px.bar(obesity_count, x="Classe", y="Total", title="Distribuição das classes"), use_container_width=True)
    with col2:
        st.plotly_chart(px.box(df, x="Obesity", y="BMI", title="IMC por classe"), use_container_width=True)
    col3, col4 = st.columns(2)
    with col3:
        fam = pd.crosstab(df["family_history"], df["Obesity"], normalize="index").reset_index().melt(id_vars="family_history", var_name="Classe", value_name="Proporção")
        st.plotly_chart(px.bar(fam, x="family_history", y="Proporção", color="Classe", barmode="stack", title="Histórico familiar e classe de peso"), use_container_width=True)
    with col4:
        activity = pd.crosstab(df["FAF_cat"], df["Obesity"], normalize="index").reset_index().melt(id_vars="FAF_cat", var_name="Classe", value_name="Proporção")
        st.plotly_chart(px.bar(activity, x="FAF_cat", y="Proporção", color="Classe", barmode="stack", title="Atividade física e classe de peso"), use_container_width=True)
    st.markdown("""
    **Insights para a equipe médica**

    A distribuição do IMC acompanha de forma consistente a progressão das classes de peso. Histórico familiar, consumo frequente de alimentos calóricos e menor frequência de atividade física aparecem como variáveis úteis para direcionar perguntas de triagem. A ferramenta pode apoiar a priorização de atendimentos, mas a decisão final deve considerar anamnese, exames e julgamento clínico.
    """)

if page == "Desempenho do modelo":
    st.subheader("Desempenho experimental")
    st.metric("Modelo selecionado", metrics["best_model"])
    st.metric("Acurácia no teste", f"{metrics['test_accuracy']:.1%}")
    st.metric("F1 macro no teste", f"{metrics['test_f1_macro']:.1%}")
    results = pd.DataFrame(metrics["results"])
    st.dataframe(results, use_container_width=True)
    report_path = ROOT / "reports" / "classification_report.csv"
    matrix_path = ROOT / "reports" / "confusion_matrix.csv"
    st.subheader("Relatório por classe")
    st.dataframe(pd.read_csv(report_path, index_col=0), use_container_width=True)
    st.subheader("Matriz de confusão")
    st.dataframe(pd.read_csv(matrix_path, index_col=0), use_container_width=True)
