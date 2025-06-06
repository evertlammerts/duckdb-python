from typing import Dict

SPECIAL: FunctionNullHandling
DEFAULT: FunctionNullHandling

NATIVE: PythonUDFType
ARROW: PythonUDFType

class FunctionNullHandling:
    DEFAULT: FunctionNullHandling
    SPECIAL: FunctionNullHandling
    def __int__(self) -> int: ...
    def __index__(self) -> int: ...
    @property
    def __members__(self) -> Dict[str, FunctionNullHandling]: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class PythonUDFType:
    NATIVE: PythonUDFType
    ARROW: PythonUDFType
    def __int__(self) -> int: ...
    def __index__(self) -> int: ...
    @property
    def __members__(self) -> Dict[str, PythonUDFType]: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...
