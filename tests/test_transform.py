import pandas as pd
from src.transform import clean_data

def test_clean_data():
    df = pd.DataFrame({
        "a":[1,2,None],
        "b": [3,4,5]
    })
    cleaned = clean_data(df)
    assert cleaned.isnull().sum().sum()==0

