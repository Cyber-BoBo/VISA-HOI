import torch

def basic_forward(model, data, **kwargs):
    return model(data)


def multiple_input_forward(model, data, **kwargs):
    return model(*data)


def gcn_forward(model, data, **kwargs):
    """for 2G-GCN"""
    input_human_segmentation = kwargs.get('input_human_segmentation', False)
    impose_segmentation_pattern = kwargs.get('impose_segmentation_pattern', 0)
    if impose_segmentation_pattern:
        if impose_segmentation_pattern == 1:
            human_segmentation = torch.ones(data[0].size()[:-1], dtype=data[0].dtype, device=data[0].device)
        else:
            raise ValueError(f'Segmentation pattern can only be 1, not {impose_segmentation_pattern}')
    elif input_human_segmentation:
        human_segmentation = data[3]
    else:
        human_segmentation = None
    model_kwargs = {
        'x_human': data[0],
        'x_objects': data[1],
        'objects_mask': data[2],
        'human_segmentation': human_segmentation,
    }
    dataset_name = kwargs.get('dataset_name', 'cad120')
    if dataset_name == 'cad120':
        input_object_segmentation = kwargs.get('input_object_segmentation', False)
        if impose_segmentation_pattern:
            if impose_segmentation_pattern == 1:
                object_segmentation = torch.ones(data[1].size()[:-1], dtype=data[1].dtype, device=data[1].device)
            else:
                raise ValueError(f'Segmentation pattern can only be 1, not {impose_segmentation_pattern}')
        elif input_object_segmentation:
            object_segmentation = data[4]
        else:
            object_segmentation = None
        model_kwargs['objects_segmentation'] = object_segmentation
        human_human_distances = human_object_distances = object_object_distances = None
        if kwargs.get('make_attention_distance_based', False):
            human_object_distances = data[5]
            object_object_distances = data[6]
    else:
        human_human_distances = human_object_distances = object_object_distances = None
        if kwargs.get('make_attention_distance_based', False):
            human_human_distances = data[4]
            human_object_distances = data[5]
            object_object_distances = data[6]
    model_kwargs['human_human_distances'] = human_human_distances
    model_kwargs['human_object_distances'] = human_object_distances
    model_kwargs['object_object_distances'] = object_object_distances
    model_kwargs['steps_per_example'] = data[7]
    model_kwargs['inspect_model'] = kwargs.get('inspect_model', False)
    return model(**model_kwargs)


def git_forward(model, data, **kwargs):
    """For GitHOI"""
    input_human_segmentation = kwargs.get('input_human_segmentation', False)
    impose_segmentation_pattern = kwargs.get('impose_segmentation_pattern', 0)
    if impose_segmentation_pattern:
        if impose_segmentation_pattern == 1:
            human_segmentation = torch.ones(data[0].size()[:-1], dtype=data[0].dtype, device=data[0].device)
        else:
            raise ValueError(f'Segmentation pattern can only be 1, not {impose_segmentation_pattern}')
    elif input_human_segmentation:
        human_segmentation = data[3]
    else:
        human_segmentation = None
    model_kwargs = {
        'x_human': data[0],
        'x_objects': data[1],
        'objects_mask': data[2],
        'human_segmentation': human_segmentation,
    }
    dataset_name = kwargs.get('dataset_name', 'cad120')
    if dataset_name == 'cad120':
        input_object_segmentation = kwargs.get('input_object_segmentation', False)
        if impose_segmentation_pattern:
            if impose_segmentation_pattern == 1:
                object_segmentation = torch.ones(data[1].size()[:-1], dtype=data[1].dtype, device=data[1].device)
            else:
                raise ValueError(f'Segmentation pattern can only be 1, not {impose_segmentation_pattern}')
        elif input_object_segmentation:
            object_segmentation = data[4]
        else:
            object_segmentation = None
        model_kwargs['objects_segmentation'] = object_segmentation
        human_human_distances = human_object_distances = object_object_distances = None
        if kwargs.get('make_attention_distance_based', False):
            human_object_distances = data[5]
            object_object_distances = data[6]
    else:
        human_human_distances = human_object_distances = object_object_distances = None
        if kwargs.get('make_attention_distance_based', False):
            human_human_distances = data[4]
            human_object_distances = data[5]
            object_object_distances = data[6]
    if kwargs.get('input_caption_action_match', False):
        caption_feature = data[8]
        action_feature = data[9]
    else:
        caption_feature = None
        action_feature = None
    model_kwargs['human_human_distances'] = human_human_distances
    model_kwargs['human_object_distances'] = human_object_distances
    model_kwargs['object_object_distances'] = object_object_distances
    model_kwargs['action_feature'] = action_feature
    model_kwargs['caption_feature'] = caption_feature
    model_kwargs['steps_per_example'] = data[7]
    model_kwargs['inspect_model'] = kwargs.get('inspect_model', False)
    return model(**model_kwargs)