import logging


def initialize_logger():
    FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger('topic_model')
    logger.setLevel(logging.INFO)
    return logger


def print_to_log(allmodules):
    for m in allmodules:
        logger.info(m.__name__ + "module loaded")
    logger.info("topic model started")


logger = initialize_logger()
