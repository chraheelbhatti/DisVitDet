import logging

def setup_logger():
    logger = logging.getLogger('DisViTDet')
    logger.setLevel(logging.DEBUG)
    # Setup logger configurations
    return logger
