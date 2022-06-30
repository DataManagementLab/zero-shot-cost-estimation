from enum import Enum


class Datatype(Enum):
    INT = 'int'
    FLOAT = 'float'
    CATEGORICAL = 'categorical'
    STRING = 'string'
    MISC = 'misc'

    def __str__(self):
        return self.value
