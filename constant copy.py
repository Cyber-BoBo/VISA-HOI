from vhoi.models.BimanualBaseline import BimanualBaseline
from vhoi.models.CAD120Baseline import CAD120Baseline

from vhoi.models.TGGCN import TGGCN

from vhoi.models.PATHoi import PATHoi




model_name_to_class_definition = {
    'bimanual_baseline': BimanualBaseline,
    'cad120_baseline': CAD120Baseline,
    'PATHoi': PATHoi,
}




MPHOI_72_SUBJECT = {'Subject12', 'Subject13', 'Subject14','Subject23','Subject25','Subject34','Subject35','Subject45'}
MPHOI_72_TASK = {'task_1_cheering', 'task_2_hair_cutting', 'task_3_co_working'}
MPHOI_72_TAKE = {'take_0', 'take_1', 'take_2'}

MPHOI_72_ACTION = ['sit', 'approach', 'retreat', 'lift', 'place',
                    'pour', 'drink' ,'cheers', 'cut', 'dry', 'work',
                    'ask', 'solve']


BIMACS_SUBJECT = {'subject_1','subject_2','subject_3','subject_4','subject_5','subject_6'}
BIMACS_TASK = {'task_1_k_cooking', 'task_2_k_cooking_with_bowls', 'task_3_k_pouring',
               'task_4_k_wiping', 'task_5_k_cereals', 'task_6_w_hard_drive',
               'task_7_w_free_hard_drive', 'task_8_w_hammering', 'task_9_w_sawing'}
BIMACS_TAKE = {'take_0', 'take_1', 'take_2', 'take_3', 'take_4', 'take_5', 'take_6', 'take_7', 'take_8', 'take_9'}
