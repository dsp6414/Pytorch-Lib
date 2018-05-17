
import numpy as np
import collections
import torch
from torch.autograd import Variable
from torch.serialization import default_restore_location



def average_checkpoints(inputs): #权值平均
    """Loads checkpoints from inputs and returns a model with averaged weights.
    Args:
      inputs: An iterable of string paths of checkpoints to load from.
    Returns:
      A dict of string keys mapping to various values. The 'model' key
      from the returned dict should correspond to an OrderedDict mapping
      string parameter names to torch Tensors.
    """
    params_dict = collections.OrderedDict()
    params_keys = None
    new_state = None
    for f in inputs:
        state = torch.load(
            f,
            map_location=(
                lambda s, _: torch.serialization.default_restore_location(s, 'cpu')
            ),
        )
        # Copies over the settings from the first checkpoint
        if new_state is None:
            new_state = state

        model_params = state['model'] #注意model关键字key

        model_params_keys = list(model_params.keys())
        if params_keys is None:
            params_keys = model_params_keys
        elif params_keys != model_params_keys:
            raise KeyError(
                'For checkpoint {}, expected list of params: {}, '
                'but found: {}'.format(f, params_keys, model_params_keys)
            )

        for k in params_keys:
            if k not in params_dict:
                params_dict[k] = []
            params_dict[k].append(model_params[k])

    averaged_params = collections.OrderedDict()
    # v should be a list of torch Tensor.
    for k, v in params_dict.items():
        summed_v = None
        for x in v:
            summed_v = summed_v + x if summed_v is not None else x
        averaged_params[k] = summed_v / len(v)
    new_state['model'] = averaged_params
    return new_state



