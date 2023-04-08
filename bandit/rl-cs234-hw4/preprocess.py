import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)

df = pd.read_csv('./data/warfarin.csv')
print(df.shape)

# you may drop and ignore the remaining 173 patients for the purpose of this question
# print(df.columns.values)
print(df['Therapeutic Dose of Warfarin'].isna().sum())
df.dropna(subset=['Therapeutic Dose of Warfarin'], inplace=True)
print(df.shape)

# drop unnecessary columns
to_drop = [col for col in df.columns.values if 'unnamed' in col.lower()]
print(to_drop)

df.drop(to_drop, axis=1, inplace=True)
print(df.shape)

df.to_csv('./data/warfarin_.csv', index=False)