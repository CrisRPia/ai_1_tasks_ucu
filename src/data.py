from pathlib import Path
from typing import Literal
import pandas as pd

DatasetName = Literal["HousingData.csv"]

data_dir = Path(__file__).parent / "../data/"

def get_data(dataset: DatasetName) -> pd.DataFrame:
    return pd.read_csv(data_dir / dataset)

train: pd.DataFrame = pd.read_csv(Path(__file__).parent / "../data/train.csv")
