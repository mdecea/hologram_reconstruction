# Holds all the error classes

class Error(Exception):
    """Base class for exceptions in this module."""
    pass

class BadImage(Exception):
    def __init__(self, message=""):
        self.message = message

    def __str__(self):
        return "Bad image: {}".format(self.message)

class LoadError(Exception):
    def __init__(self, filename, message):
        super().__init__("Error loading file %r: %s" % (filename, message))

class MissingParameter(Exception):
    def __init__(self, parameter_name):
        self.parameter_name = parameter_name
    def __str__(self):
        return ("Calculation requires specification of " + self.parameter_name + ".")