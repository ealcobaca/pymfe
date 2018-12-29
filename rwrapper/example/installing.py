from pymfe.utils.install import Install

install_by_devtools = False
if install_by_devtools == True:
    Install.devtoolsR()
    Install.mfeR(source="devtools")
    Install.ecolR(source="devtools")
    Install.installR(["rivolli/mfe", "SmartDataAnalytics/ECoL"], source="devtools")
else:
    Install.mfeR(source="cran")
    Install.ecolR(source="cran")
    Install.installR(["mfe", "ECoL"], source="cran")

