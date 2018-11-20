class MfeError(Exception):
    """ Basic exception for errors raised by Mfe """
    def __init__(self, msg):
        super(MfeError, self).__init__("Something is wrong: ...\n" + msg)

class RLibNotFound(MfeError):
    """ When some R library is not found """
    def __init__(self, lib):
        super(RLibNotFound, self).__init__(
            """
            The library %s was not found.
            Please check if this library was installed.
            You can use pymfe.util.check() and pymfe.util.install()
            """ % lib
        )
        self. lib = lib

class MfeOption(MfeError):
    """ When some option is wrong """
    def __init__(self, var, option, options):
        super(MfeOption, self).__init__(
            """
            Please check the correct option.
            You set '{0}' with '{1}', but the possible options are:
            '{2}'.
            """.format(var, option, options)
        )
        self.option = option
        self.options = options

