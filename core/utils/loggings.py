import logging,sys
import inspect

def setup_logger(debug_filename = None, output_filename = None, level=logging.INFO):
    logger = logging.getLogger("futuremat")
    logger.setLevel(level)
    # create the logging file handler

    FORMAT = "[%(asctime)s - %(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"

    formatter = logging.Formatter(FORMAT)

    if debug_filename != None:
        fh = logging.FileHandler(debug_filename, mode = 'w+')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    if output_filename != None:
        log_handler = logging.FileHandler(output_filename, mode = 'w+')
        log_handler.setFormatter(formatter)
        logger.addHandler(log_handler)
    else:
        log_handler = logging.StreamHandler(sys.stdout)
        log_handler.setFormatter(formatter)
        logger.addHandler(log_handler)
    log_handler.level = level

    return logger
