import numpy as np
import pandas as pd

df_pred=pd.read_csv(r'C:\Users\yylc0\Desktop\arbeite\TCN_project\project\outputs\inference_predictions.csv')
df_actual=pd.read_csv(r'C:\Users\yylc0\Desktop\arbeite\TCN_project\project\outputs\inference_actual_returns.csv')


print(df_pred['000852.SH'].tail(10))
df_pred=df_pred.shift(1)
print(df_pred['000852.SH'].tail(10))


















