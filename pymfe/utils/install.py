from rpy2.robjects.vectors import StrVector
from rpy2.robjects import r
from rpy2.robjects.packages import isinstalled, importr
from pymfe.exceptions.exceptions import MfeOption
from pymfe.exceptions.exceptions import RLibNotFound, MfeOption


class Install(object):
    """ """
    __devtools_cran = "devtools"
    __mfe_cran = "mfe"
    __ecol_cran = "ECoL"
    __mfe_devtools = "rivolli/mfe"
    __ecol_devtools = "SmartDataAnalytics/ECoL"
    __source_opt = {"cran", "devtools"}
    __mirror_opt = "`mirror` should be > 0, see more information here --> https://cran.r-project.org/mirrors.html"


    @classmethod
    def get_mirror(cls):
        return cls.__mirror_opt

    @classmethod
    def get_all_source(cls):
        return list(cls.__source_opt)


    @classmethod
    def devtoolsR(cls, mirror=1):
        package = cls.__devtools_cran
        cls.installR(package, "cran", mirror)


    @classmethod
    def mfeR(cls, source="cran", mirror=1):
        package = None
        if source == "cran":
            package = cls.__mfe_cran
        elif source == "devtools":
            package = cls.__mfe_devtools

        cls.installR(package, source, mirror)


    @classmethod
    def ecolR(cls, source="cran", mirror=1):
        package = None
        if source == "cran":
            package = cls.__ecol_cran
        elif source == "devtools":
            package = cls.__ecol_devtools

        cls.installR(package, source, mirror)


    @classmethod
    def installedR(cls, package):
        return isinstalled(package)


    @classmethod
    def installR(cls, package, source="cran", mirror=1):
        if source == "cran":
            utils = importr('utils')
            utils.chooseCRANmirror(ind=mirror)

            if isinstance(package, list):
                utils.install_packages(StrVector(package))
            else:
                utils.install_packages(package)

        elif source == "devtools":
            # verify if devtools is instelled
            exist = isinstalled(cls.__devtools_cran)
            if exist == True:
                devtools = importr(cls.__devtools_cran)
            else:
                raise RLibNotFound(cls.__devtools_cran)

            if isinstance(package, list):
                r("devtools::install_github({0})".format(StrVector(package).r_repr()))
            else:
                r("devtools::install_github(\'{0}\')".format(package))

        else:
            raise MfeOption("source", source, cls.get_all_source())
