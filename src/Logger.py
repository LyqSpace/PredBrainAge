import logging
import logging.handlers


class Logger:

    def __init__(self, event_name, log_file):

        fmt = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
        formatter = logging.Formatter(fmt)
        handler = logging.handlers.RotatingFileHandler(log_file)
        handler.setFormatter(formatter)
        self._logger = logging.getLogger(event_name)
        self._logger.addHandler(handler)
        self._logger.setLevel(logging.INFO)

    def log(self, message):
        self._logger.info(message)