import pandas as pd
import os

local_dir = "/gpfs/radev/project/xu_hua/xp83/OCR_Task"
gr_fp = "OCR_DATA/local_file_version/GreekOCR_v1.parquet"
df = pd.read_parquet(os.path.join(local_dir, gr_fp))

i=5
# print(df.columns)
# print(df.iloc[i,2])
# print(df.iloc[i,-2])

en_fp = "OCR_DATA/local_file_version/EnglishOCR_v2.parquet"
df = pd.read_parquet(os.path.join(local_dir, en_fp))

i=702
print(df.columns)
print(df.iloc[i,2])
print(df.iloc[i,-2])