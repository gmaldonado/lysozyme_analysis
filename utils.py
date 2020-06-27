import pandas as pd

def read_dataset(path,sheet):
    xls_file = pd.ExcelFile(path)
    dataset = pd.read_excel(xls_file,sheet)
    dataset = dataset.loc[:, ~dataset.columns.str.contains('^Unnamed')]
    return dataset