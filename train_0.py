import hydra
from omegaconf import DictConfig
import torch

from pyrutils.torch.train_utils import train, save_checkpoint
from pyrutils.torch.multi_task import MultiTaskLossLearner
# from vhoi.data_loading import load_training_data, select_model_data_feeder, select_model_data_fetcher
# from vhoi.data_loading import determine_num_classes
from vhoi.my_data_loading import load_training_data, select_model_data_feeder, select_model_data_fetcher
from vhoi.my_data_loading import determine_num_classes

from vhoi.losses import select_loss, decide_num_main_losses, select_loss_types, select_loss_learning_mask
from vhoi.models import select_model, load_model_weights
from vhoi.complexity import model_param_summary


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    seed = 42
    torch.set_num_threads(cfg.resources.num_threads)        # 设置CPU多线程运算线程数
    # Data
    # 相关数据设置
    model_name, model_input_type = cfg.models.metadata.model_name, cfg.models.metadata.input_type     # 模型名称和输入类型
    batch_size, val_fraction = cfg.models.optimization.batch_size, cfg.models.optimization.val_fraction       # batch size设置val fraction设置
    misc_dict = cfg.models.get('misc', default_value={})
    sigma = misc_dict.get('segmentation_loss', {}).get('sigma', 0.0)    # 分段损失函数
    train_loader, val_loader, data_info, scalers = load_training_data(cfg.data.data, model_name, model_input_type,
                                                                      batch_size=batch_size,
                                                                      val_fraction=val_fraction,
                                                                      seed=seed, debug=False, sigma=sigma)
    # Model
    Model = select_model(model_name)    # 选择相应的模型，输入是模型名称['bimanual_baseline','cad120_baseline','2G-GCN']
    model_creation_args = cfg.models.parameters        # 模型相关参数，详见models文件夹中.yaml文档
    model_creation_args = {**data_info, **model_creation_args}          # data_info: Input的size
    dataset_name = cfg.data.data.name                            
    num_classes = determine_num_classes(model_name, model_input_type, dataset_name) # 数据集分类数
    model_creation_args['num_classes'] = num_classes
    device = 'cuda:0' if torch.cuda.is_available() and cfg.resources.use_gpu else 'cpu'   # 使用GPU或CPU
    model = Model(**model_creation_args).to(device)
    model_param_summary(model)
    print(misc_dict)
    if misc_dict.get('pretrained', False) and misc_dict.get('pretrained_path') is not None:
        """get(key,value);当字典中存在key时,则不会赋值;不存在时,则赋值,可以将value理解为默认值"""
        state_dict = load_model_weights(misc_dict['pretrained_path'])
        model.load_state_dict(state_dict, strict=False)         # 从state_dict中将参数和缓冲拷贝到当前这个模块及其子模块中
        print('预训练权重成功加载')
    params = model.parameters()         
    optimizer = torch.optim.Adam(params, lr=cfg.models.optimization.learning_rate)
    criterion, loss_names = select_loss(model_name, model_input_type, dataset_name, cfg=cfg)
    mtll_model = None
    if misc_dict.get('multi_task_loss_learner', False):
        loss_types = select_loss_types(model_name, dataset_name, cfg=cfg)
        mask = select_loss_learning_mask(model_name, dataset_name, cfg=cfg)
        mtll_model = MultiTaskLossLearner(loss_types=loss_types, mask=mask).to(device)
        optimizer.add_param_group({'params': mtll_model.parameters()})
    # Some config + model training
    tensorboard_log_dir = cfg.models.logging.root_log_dir
    checkpoint_name = cfg.models.logging.checkpoint_name
    fetch_model_data = select_model_data_fetcher(model_name, model_input_type,
                                                 dataset_name=dataset_name, **{**misc_dict, **cfg.models.parameters})
    feed_model_data = select_model_data_feeder(model_name, model_input_type, dataset_name=dataset_name, **misc_dict)
    num_main_losses = decide_num_main_losses(model_name, dataset_name, {**misc_dict, **cfg.models.parameters})
    checkpoint = train(model, train_loader, optimizer, criterion, cfg.models.optimization.epochs, device, loss_names,
                       clip_gradient_at=cfg.models.optimization.clip_gradient_at,
                       fetch_model_data=fetch_model_data, feed_model_data=feed_model_data,
                       val_loader=val_loader, mtll_model=mtll_model, num_main_losses=num_main_losses,
                       tensorboard_log_dir=tensorboard_log_dir, checkpoint_name=checkpoint_name)
    # Logging
    if (log_dir := cfg.models.logging.log_dir) is not None:
        checkpoint['scalers'] = scalers
        save_checkpoint(log_dir, checkpoint, checkpoint_name=checkpoint_name, include_timestamp=False)


if __name__ == '__main__':
    main()
