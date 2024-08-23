import logging
import colorlog


class LoggingConfiguration:
    """Logging configuration for the base logger

    Establishes the handlers for the logger with the name "diamond_assignment"
    """

    def __init__(self, level: str = "INFO"):
        """Initialize the base logger

        :param level: The logging level for the StreamHandler.
            Should be DEBUG, INFO, WARNING, CRITICAL or ERROR.
        """
        self.logger = logging.getLogger("diamond_assignment")
        self.logger.setLevel("DEBUG")
        self.handler = colorlog.StreamHandler()
        self.handler.setLevel(level)
        self.handler.setFormatter(
            colorlog.ColoredFormatter(
                "%(log_color)" "s%(levelname)" "s:%(name)" "s:%(message)s"
            )
        )
        self.logger.addHandler(self.handler)
