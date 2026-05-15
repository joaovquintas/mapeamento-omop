import pandas as pd

# Ler o CSV
df = pd.read_csv("datasets/medical_device_omop_consolidated.csv")

# Pegar as 1001 primeiras linhas
df_1001 = df.head(1000)

# Salvar em um novo CSV
df_1001.to_csv("datasets/medical_device_dataset.csv", index=False)