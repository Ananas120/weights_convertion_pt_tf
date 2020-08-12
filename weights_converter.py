import numpy as np

def transpose_weights(weights):
    if len(weights.shape) <= 1:
        return weights
    if len(weights.shape) == 2:
        return weights.T
    if len(weights.shape) == 3:
        return np.transpose(weights, [2, 1, 0])
    else:
        raise ValueError("Unknown weights shape : {}".format(weights.shape))

""" Pytorch to Tensorflow convertion """

def get_pt_layers(pt_model):
    layers = {}
    for k, v in pt_model.state_dict().items():
        layer_name = '.'.join(k.split('.')[:-1])
        if layer_name not in layers: layers[layer_name] = []
        layers[layer_name].append(v.cpu().numpy())
    return layers

def pt_convert_layer_weights(layer_weights):
    new_weights = []
    if len(layer_weights) < 4:
        new_weights = layer_weights
    elif len(layer_weights) == 4:
        new_weights = layer_weights[:2] + [layer_weights[2] + layer_weights[3]]
    elif len(layer_weights) == 5:
        new_weights = layer_weights[:4]
    elif len(layer_weights) == 8:
        new_weights = layer_weights[:2] + [layer_weights[2] + layer_weights[3]]
        new_weights += layer_weights[4:6] + [layer_weights[6] + layer_weights[7]]
    else:
        raise ValueError("Unknown weights length : {}\n  Shapes : {}".format(len(layer_weights), [tuple(v.shape) for v in layer_weights]))
    
    return [transpose_weights(w) for w in new_weights]

def pt_convert_model_weights(pt_model, tf_model, verbose = False):
    pt_layers = get_pt_layers(pt_model)
    converted_weights = []
    for layer_name, layer_variables in pt_layers.items():
        converted_variables = pt_convert_layer_weights(layer_variables) if 'embedding' not in layer_name else layer_variables
        converted_weights += converted_variables
        
        if verbose:
            print("Layer : {} \t {} \t {}".format(
                layer_name, 
                [tuple(v.shape) for v in layer_variables],
                [tuple(v.shape) for v in converted_variables],
            ))
    
    tf_model.set_weights(converted_weights)
    print("Weights converted successfully !")
    
    
""" Tensorflow to Pytorch converter """

def get_tf_layers(tf_model):
    layers = {}
    for v in tf_model.variables:
        layer_name = '/'.join(v.name.split('/')[:-1])
        if layer_name not in layers: layers[layer_name] = []
        layers[layer_name].append(v.numpy())
    return layers

def tf_convert_layer_weights(layer_weights):
    new_weights = []
    if len(layer_weights) < 3 or len(layer_weights) == 4:
        new_weights = layer_weights
    elif len(layer_weights) == 3:
        new_weights = layer_weights[:2] + [layer_weights[2] / 2., layer_weights[2] / 2.]
    else:
        raise ValueError("Unknown weights length : {}\n  Shapes : {}".format(len(layer_weights), [tuple(v.shape) for v in layer_weights]))
    
    return [transpose_weights(w) for w in new_weights]


def tf_convert_model_weights(tf_model, pt_model, verbose = False):
    import torch
    
    pt_layers = pt_model.state_dict()
    tf_layers = get_tf_layers(tf_model)
    converted_weights = []
    for layer_name, layer_variables in tf_layers.items():
        converted_variables = tf_convert_layer_weights(layer_variables) if 'embedding' not in layer_name else layer_variables
        converted_weights += converted_variables
        
        if verbose:
            print("Layer : {} \t {} \t {}".format(
                layer_name, 
                [tuple(v.shape) for v in layer_variables],
                [tuple(v.shape) for v in converted_variables],
            ))
    
    tf_idx = 0
    for i, (pt_name, pt_weights) in enumerate(pt_layers.items()):
        if len(pt_weights.shape) == 0: continue
        
        pt_weights.data = torch.from_numpy(converted_weights[tf_idx])
        tf_idx += 1
    
    pt_model.load_state_dict(pt_layers)
    print("Weights converted successfully !")

""" Partial transfert learning """

def transfer_weights(target_model, pretrained_model):
    new_weights = []
    target_variables = target_model.variables
    pretrained_variables = pretrained_model.variables
    for i in range(len(target_variables)):
        if len(target_variables[i].shape) != len(pretrained_variables[i].shape):
            raise ValueError("Le nombre de dimension des variables {} est différent !\n  Target shape : {}\n  Pretrained shape : {}".format(i, target_variables[i].shape, pretrained_variables[i].shape))
        
        new_v = target_variables[i].numpy()
        v = pretrained_variables[i].numpy()
        if new_v.ndim == 1:
            max_0 = min(new_v.shape[0], v.shape[0])
            new_v[:max_0] = v[:max_0]
        elif new_v.ndim == 2:
            max_0 = min(new_v.shape[0], v.shape[0])
            max_1 = min(new_v.shape[1], v.shape[1])
            new_v[:max_0, :max_1] = v[:max_0, :max_1]
        elif new_v.ndim == 3:
            max_0 = min(new_v.shape[0], v.shape[0])
            max_1 = min(new_v.shape[1], v.shape[1])
            max_2 = min(new_v.shape[2], v.shape[2])
            new_v[:max_0, :max_1, :max_2] = v[:max_0, :max_1, :max_2]
        elif new_v.ndim == 4:
            max_0 = min(new_v.shape[0], v.shape[0])
            max_1 = min(new_v.shape[1], v.shape[1])
            max_2 = min(new_v.shape[2], v.shape[2])
            max_3 = min(new_v.shape[3], v.shape[3])
            new_v[:max_0, :max_1, :max_2, :max_3] = v[:max_0, :max_1, :max_2, :max_3]
        else:
            raise ValueError("Variable dims > 4 non géré !")
        new_weights.append(new_v)
    
    target_model.set_weights(new_weights)
    print("Weights transfered successfully !")
        