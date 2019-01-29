import logging
import logging.config
from tool.logging_conf import PINGAN_LOGGING_CONF

logging.config.dictConfig(PINGAN_LOGGING_CONF)
logger = logging.getLogger("pingan")
