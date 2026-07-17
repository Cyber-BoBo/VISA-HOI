import os
import torch

from .BimanualBaseline import BimanualBaseline
from .CAD120Baseline import CAD120Baseline
from .TGGCN import TGGCN
from .VISAHoi import VISAhoi

""" 选择对应的模型 """
def select_model(model_name: str):
    model_name_to_class_definition = {
        'bimanual_baseline': BimanualBaseline,
        'cad120_baseline': CAD120Baseline,
        '2G-GCN': TGGCN,
        'VISA-HOI': VISAHoi
    }
    return model_name_to_class_definition[model_name]


def load_model_weights(model_dir: str):
    checkpoint_file = os.path.join(model_dir, os.path.basename(model_dir) + '.tar')
    checkpoint = torch.load(checkpoint_file)
    state_dict = checkpoint['model_state_dict']
    return state_dict