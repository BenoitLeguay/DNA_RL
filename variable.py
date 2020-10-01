import torch
import logging
import os
import a2c

logger = None
PATH = str(os.path.dirname(os.path.realpath(__file__)))
PROCESSING_LOGS = os.path.join(PATH, "logs/")
if not os.path.isdir(PROCESSING_LOGS):
    os.makedirs(PROCESSING_LOGS)
LOG_INFO = os.path.join(PROCESSING_LOGS, 'info.log')

actor_types = {'actor_rnn': a2c.Actor, 'actor_vanilla': a2c.ActorVanilla, 'actor_n_actions': a2c.NActionActor}


def init_logger():
    """
    Init the logger object used to put log in stderr

    :return:
    """
    global logger

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    log_formatter = logging.Formatter("%(asctime)s [%(process)d][%(levelname)-5.5s]  %(message)s")

    info_handler = logging.FileHandler(LOG_INFO)
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(log_formatter)
    logger.addHandler(info_handler)


oracle_checkpoint_path = '/home/benoit/Documents/work/GA_GEN/RalucaPred/lightning_logs/version_5/checkpoints/epoch=45.ckpt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')