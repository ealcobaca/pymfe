from pymfe.mfe.mfer import MetaFeatureExtractorR
import pandas as pd

df = pd.read_csv("example/datasets/dataset_61_iris.csv")
X = df.iloc[:, 0:4]
y = df.iloc[:, 4]

mfer = MetaFeatureExtractorR()
mf = mfer.extract(X,y)
print(mf)

mfer = MetaFeatureExtractorR(MetaFeatureExtractorR.get_all_groups(),
                             MetaFeatureExtractorR.get_all_summary())
mf = mfer.extract(X,y)
print(mf)

