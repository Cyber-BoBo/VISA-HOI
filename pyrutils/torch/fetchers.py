def single_input_single_output(dataset, device):
    data, target = dataset[0].to(device), dataset[1].to(device)
    return data, target


def single_input_multiple_output(dataset, device):
    data, targets = dataset[0], dataset[1:]
    data, targets = data.to(device), [target.to(device) for target in targets]
    return data, targets


def multiple_input_multiple_output(dataset, device, n):
    data, targets = dataset[:n], dataset[n:]
    data, targets = [datum.to(device) for datum in data], [target.to(device) for target in targets]
    return data, targets


def gcn_fetcher(dataset, device, **kwargs):
    """for 2G-GCN"""
    data = []
    data.append(dataset[0].to(device))
    data.append(dataset[1].to(device))
    data.append(dataset[2].to(device))
    if kwargs.get('input_human_segmentation', False):
        data.append(dataset[3].to(device))
    else:
        data.append(dataset[3])
    dataset_name = kwargs.get('dataset_name', 'cad120')
    if dataset_name == 'cad120':
        if kwargs.get('input_object_segmentation', False):
            data.append(dataset[4].to(device))
        else:
            data.append(dataset[4])
        if kwargs.get('make_attention_distance_based', False):
            data.append(dataset[5].to(device))
            data.append(dataset[6].to(device))
        else:
            data.append(dataset[5])
            data.append(dataset[6])
        targets = [target.to(device) for target in dataset[10:]]
    else:  # bimanual & mphoi
        if kwargs.get('make_attention_distance_based', False):
            data.append(dataset[4].to(device))
            data.append(dataset[5].to(device))
            data.append(dataset[6].to(device))
        else:
            data.append(dataset[4])
            data.append(dataset[5])
            data.append(dataset[6])
        targets = [target.to(device) for target in dataset[10:]]
    data.append(dataset[7].to(device))
    return data, targets



def git_fetcher(dataset, device, **kwargs):
    """for GitHOI"""
    data = []
    data.append(dataset[0].to(device))
    data.append(dataset[1].to(device))
    data.append(dataset[2].to(device))
    if kwargs.get('input_human_segmentation', False):
        data.append(dataset[3].to(device))
    else:
        data.append(dataset[3])
    dataset_name = kwargs.get('dataset_name', 'cad120')
    if dataset_name == 'cad120':
        if kwargs.get('input_object_segmentation', False):
            data.append(dataset[4].to(device))
        else:
            data.append(dataset[4])
        if kwargs.get('make_attention_distance_based', False):
            data.append(dataset[5].to(device))
            data.append(dataset[6].to(device))
        else:
            data.append(dataset[5])
            data.append(dataset[6])
        targets = [target.to(device) for target in dataset[10:]]
    else:  # bimanual & mphoi
        if kwargs.get('make_attention_distance_based', False):
            data.append(dataset[4].to(device))
            data.append(dataset[5].to(device))
            data.append(dataset[6].to(device))
        else:
            data.append(dataset[4])
            data.append(dataset[5])
            data.append(dataset[6])
        targets = [target.to(device) for target in dataset[10:]]
    data.append(dataset[7].to(device))
    if kwargs.get('input_caption_action_match', False):
        data.append(dataset[8].to(device))
        data.append(dataset[9].to(device))
    return data, targets