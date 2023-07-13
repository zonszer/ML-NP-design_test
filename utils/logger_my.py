import logging
import os

class Logger():
    COLOR_CODES = {
        'HEADER': "\033[95m",
        'OKBLUE': "\033[94m",
        'OKGREEN': "\033[92m",
        'WARNING': "\033[93m",
        'FAIL': "\033[91m",
        'END': "\033[0m",
        'BOLD': "\033[1m",
        'UNDERLINE': "\033[4m",
        'BLACK': "\033[1;30m",
        'RED': "\033[1;31m",
        'GREEN': "\033[1;32m",
        'YELLOW': "\033[1;33m",
        'BLUE': "\033[1;34m",
        'PURPLE': "\033[1;35m",
        'CYAN': "\033[1;36m",
        'WHITE': "\033[1;37m",
    }

    class ColorfulStreamHandler(logging.StreamHandler):
        """A custom StreamHandler class that inherits from logging.StreamHandler."""
        def emit(self, record):
            try:
                message = self.format(record)
                if record.__dict__.get('color'):
                    stream = self.stream
                    stream.write(Logger.COLOR_CODES[record.__dict__.get('color')] + message + Logger.COLOR_CODES['END'] + self.terminator)
                else:
                    stream = self.stream
                    stream.write(message + self.terminator)
                self.flush()
            except Exception:
                self.handleError(record)

    def __init__(self, log_path=None):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        if log_path:
            self._init(log_path)

    def _init(self, log_path):
        self.savepath = log_path
        if not self.logger.handlers:
            # Logging to a file
            if not os.path.exists(log_path):
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, 'a'): pass
            file_handler = logging.FileHandler(log_path, encoding="utf-8", mode="a")
            file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
            self.logger.addHandler(file_handler)

            # Logging to console
            stream_handler = self.ColorfulStreamHandler()
            stream_handler.setFormatter(logging.Formatter('%(message)s'))
            self.logger.addHandler(stream_handler)

    def log(self, level, message, color=None):
        """
        Log a message with the specified level and color.

        :param level: Logging level (e.g., logging.INFO, logging.WARNING, logging.ERROR)
        :param message: The message to log
        :param color: The color to use for the message (optional)
        """
        self.logger.log(level, message, extra={'color': color})

    def colorize(self, message, color):
        """
        Add color codes to the message.

        :param message: The message to colorize
        :param color: The color to use
        :return: The colorized message
        """
        return self.COLOR_CODES[color] + message + self.COLOR_CODES['END']

