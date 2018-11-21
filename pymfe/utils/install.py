from rpy2.robjects.vectors import StrVector
from rpy2.robjects import r
from rpy2.robjects.packages import importr
from pymfe.exceptions.exceptions import MfeOption


class Install(object):
    """ """
    __devtools_cran = "devtools"
    __mfe_cran = "mfe"
    __ecol_cran = "EcoL"
    __mfe_devtools = "rivolli/mfe"
    __ecol_devtools = "SmartDataAnalytics/ECoL"
    __devtools = """ if (!require("devtools")) install.packages("devtools");
                 devtools::install_github("{0}");
                 """
    __source_opt = {"cran", "devtools"}
    __mirror_opt = "`mirror` should be > 0, see more information here --> https://cran.r-project.org/mirrors.html"


    @classmethod
    def get_mirror(cls):
        return cls.__mirror_opt

    @classmethod
    def get_all_source(cls):
        return list(cls.__source_opt)


    @classmethod
    def devtoolsR(cls, mirror=1)
        package = cls.__devtools_cran
        cls.installr(package, cls.__source_opt[0], mirror)


    @classmethod
    def mfeR(cls, source="cran", mirror=1):
        package = None
        if source == cls.__source_opt[0]:
            package = cls.__mfe_cran
        elif source == cls.__source_opt[1]:
            package = cls.__mfe_devtools

        cls.installr(package, source, mirror)


    @classmethod
    def ecolR(cls, source="cran", mirror=1):
        package = None
        if source == cls.__source_opt[0]:
            package = cls.__ecol_cran
        elif source == cls.__source_opt[1]:
            package = cls.__ecol_devtools

        cls.installr(package, source, mirror)


    @classmethod
    def installr(cls, package, source="cran", mirror=1):
        if source == cls.__source_opt[0]:
            utils = importr('utils')
            utils.chooseCRANmirror(ind=mirror)
            utils.install_packages(StrVector(package))
        elif source == cls.__source_opt[1]:
            # verify if devtools is instelled
            exist = isinstalled(cls.__devtools_cran)
            if exist == True:
                devtools = importr(cls.__devtools_cran)
            else:
                # if devtools not found
                raise RLibNotFound(cls.__devtools_cran)
            devtools(StrVector(package))
        else:
            raise MfeOption("source", source, cls.get_all_source())
