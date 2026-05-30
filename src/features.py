import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class ObesityFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df["BMI"] = df["Weight"] / (df["Height"] ** 2)
        df["Age_Group"] = pd.cut(df["Age"], bins=[0, 20, 30, 40, 50, 70], labels=["14-20", "21-30", "31-40", "41-50", "51+"])
        df["FCVC_round"] = df["FCVC"].round().clip(1, 3).astype(int).astype(str)
        df["NCP_round"] = df["NCP"].round().clip(1, 4).astype(int).astype(str)
        df["CH2O_round"] = df["CH2O"].round().clip(1, 3).astype(int).astype(str)
        df["FAF_round"] = df["FAF"].round().clip(0, 3).astype(int).astype(str)
        df["TUE_round"] = df["TUE"].round().clip(0, 2).astype(int).astype(str)
        return df
