import inspect
from abc import ABCMeta
from enum import IntEnum as Enum
import os
from datetime import datetime


class Singleton():
    _instance = None

    def __init__(self):
        raise NotImplementedError('Call getInstance() instead')

    @classmethod
    def getInstance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            # Put any initialization here.
        return cls._instance


class LogLevel(Enum):
    DEBUG = 1
    VERBOSE = 2
    INFO = 3
    WARN = 4
    ERROR = 5


DEFAULT_LOG_LEVEL = LogLevel.VERBOSE

class ComponentLogger(Singleton, metaclass=ABCMeta):
    component_name = 'BERT_classifier'
    level = DEFAULT_LOG_LEVEL

    def log(self, *args, level=LogLevel.INFO):
        if level >= self.level:
            lines = ' '.join(map(lambda d: str(d), args))
            lines = lines.split('\n')
            try:
                func_name = inspect.stack()[1][3]
            except KeyError:
                func_name = "UNKNOWN"
            for line in lines:
                print(datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f") + ':',
                      level.name+':', self.component_name + ':',
                      func_name + ':', line)

    def set_component_name(self, component_name):
        self._instance.component_name = component_name


env_level_map = {'DEBUG': LogLevel.DEBUG, 'VERBOSE': LogLevel.VERBOSE,
                 'INFO': LogLevel.INFO, 'ERROR': LogLevel.ERROR}

component_logger = ComponentLogger.getInstance()

try:
    component_logger.level = env_level_map[
        os.environ.get("LOG_LEVEL")]
except KeyError:
    component_logger.level = DEFAULT_LOG_LEVEL
