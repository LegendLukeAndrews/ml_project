import pandas as pd
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna()
    return df