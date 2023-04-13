import glob
import os
import torch


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0-tau) + param.data * tau
        )


def save_model(model, logs_path, types, step, max_model=None):
    start = len(types) + 1
    os.makedirs(logs_path, exist_ok=True)
    if max_model is not None:
        model_list = glob.glob(os.path.join(logs_path, '*.pth'))
        if len(model_list) > max_model - 1:
            min_step = min([
                int(li.split('/')[-1][start:-4]) for li in model_list
            ])
            os.remove(
                os.path.join(logs_path, '{}-{}.pth'.format(types, min_step))
            )
    logs_path = os.path.join(logs_path, '{}-{}.pth'.format(types, step))
    torch.save(model.state_dict(), logs_path)
    return logs_path


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias
