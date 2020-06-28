import torch

oracle_checkpoint_path = '/home/benoit/Documents/work/GA_GEN/RalucaPred/lightning_logs/version_5/checkpoints/epoch=45.ckpt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')