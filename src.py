import os
import glob
import numbers

import pandas as pd
import numpy as np

from enum import Enum

class classes_enum(Enum):
    none = ''
    antrac = 'A'
    collapse = 'C'
    antrac_and_collapse = 'AC'

def get_class_from_path(filepath: str) -> classes_enum:
    basename = os.path.basename(filepath)
    name = os.path.splitext(basename)[0]
    
    fruto, n, *classes = name.split('_')

    assert fruto == 'Fruto', "first symbol is not 'Fruto'."
    assert isinstance(int(n), numbers.Integral), "second symbol is not a Integer"

    if len(classes) == 1:
        return classes_enum.antrac if classes[0] == classes_enum.antrac else classes_enum.collapse
    else:
        switch = {
            0: classes_enum.none,
            2: classes_enum.antrac_and_collapse
        }
        return switch[len(classes)]

files = glob.glob1('data', '*.xls')

classes = list(map(get_class_from_path, files))

has_antrac = [c == classes_enum.antrac or c == classes_enum.antrac_and_collapse for c in classes]
has_collapse = [c == classes_enum.collapse or c == classes_enum.antrac_and_collapse for c in classes]


data = []
for f, a, c in zip(files, has_antrac, has_collapse):
    path = os.path.join('data', f)
    print(path)
    
    df = pd.read_excel(path).drop(['λ(nm)', 'Áreas', 'Parâmetros: TI(ms), Md, Bx'], axis=1).T
    df['antrac'] = a
    df['collapse'] = c

    data.append(df)
    
final_df = pd.concat(data).reset_index(drop=True)
final_df.columns = final_df.columns.astype(str)
final_df.to_parquet('mango_spectra.parquet')