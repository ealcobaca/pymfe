
import pandas as pd
import numpy as np
from pymfe.exceptions.mfe_error import RLibNotFound, MfeOption
from rpy2.robjects.packages import isinstalled, importr
    @staticmethod
from rpy2.robjects import pandas2ri
from rpy2.robjects.vectors import StrVector


class MetaFeatureExtractorR:

    rmfe_groups = {"general", "statistical", "infotheo", "model.based", "landmarking"}
    rmfe_summary = {"min", "max", "quantile", "mean", "sd", "var", "median"}

    def __init__(self,
        group=["general", "statistical", "infotheo", "model.based", "landmarking"],
        summary=["mean", "sd"]):

        if not self.rmfe_groups.issuperset(set(group)):
            raise MfeOption("group", group, self.get_all_groups())
        self.group = StrVector(group)

        if not self.rmfe_summary.issuperset(set(summary)):
            raise MfeOption("summary", summary, self.get_all_summary())
        self.summary = StrVector(summary)


    @classmethod
    def get_all_groups(cls):
        return list(cls.rmfe_groups)


    @classmethod
    def get_all_summary(cls):
        return list(cls.rmfe_summary)


    def extract(self, X, y):
        mfe_name = "mfe"
        exist = isinstalled(mfe_name)

        if exist:
            # load if MFE library is installed
            mfe = importr(mfe_name)
        else:
            # if MFE not found
            raise RLibNotFound(mfe_name)

        # extracting meta-features from R MFE
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if not isinstance(X, pd.DataFrame):
            y = pd.DataFrame(y)

        pandas2ri.activate()
        result = mfe.metafeatures(X, y, self.group, self.summary)
        pandas2ri.deactivate()

        return pd.DataFrame(np.array(result), index=np.array(result.names))
