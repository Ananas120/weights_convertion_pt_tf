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
    state_dict = pt_model.state_dict() if not isinstance(pt_model, dict) else pt_model
    for k, v in state_dict.items():
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
    
    partial_transfert_learning(tf_model, converted_weights)
    print("Weights converted successfully !")
    
    
""" Tensorflow to Pytorch converter """

def get_tf_layers(tf_model):
    layers = {}
    variables = tf_model.variables if not isinstance(tf_model, list) else tf_model
    for v in variables:
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

def partial_transfert_learning(target_model, pretrained_model, partial_transfert = True):
    target_variables = target_model.variables if not isinstance(target_model, list) else target_model
    pretrained_variables = pretrained_model.variables if not isinstance(pretrained_model, list) else pretrained_model
    
    new_weights = []
    for i, (v, pretrained_v) in enumerate(zip(target_variables, pretrained_variables)):
        if len(v.shape) != len(pretrained_v.shape):
            raise ValueError("Le nombre de dimension des variables {} est différent !\n  Target shape : {}\n  Pretrained shape : {}".format(i, v.shape, pretrained_v.shape))
        
        v = v.numpy()
        if not isinstance(pretrained_v, np.ndarray) : pretrained_v = pretrained_v.numpy()
        
        
        if v.shape == pretrained_v.shape:
            new_weights.append(pretrained_v)
        else:
            if not partial_transfert:
                print("Variables {} shapes mismatch ({} vs {}), skipping it".format(i, v.shape, pretrained_v.shape))
                new_weights.append(v)
                continue
            
            print("Variables {} shapes mismatch ({} vs {}), making partial transfert".format(i, v.shape, pretrained_v.shape))
        
            if v.ndim == 1:
                max_0 = min(v.shape[0], pretrained_v.shape[0])
                v[:max_0] = pretrained_v[:max_0]
            elif v.ndim == 2:
                max_0 = min(v.shape[0], pretrained_v.shape[0])
                max_1 = min(v.shape[1], pretrained_v.shape[1])
                v[:max_0, :max_1] = pretrained_v[:max_0, :max_1]
            elif v.ndim == 3:
                max_0 = min(v.shape[0], pretrained_v.shape[0])
                max_1 = min(v.shape[1], pretrained_v.shape[1])
                max_2 = min(v.shape[2], pretrained_v.shape[2])
                v[:max_0, :max_1, :max_2] = pretrained_v[:max_0, :max_1, :max_2]
            elif v.ndim == 4:
                max_0 = min(v.shape[0], pretrained_v.shape[0])
                max_1 = min(v.shape[1], pretrained_v.shape[1])
                max_2 = min(v.shape[2], pretrained_v.shape[2])
                max_3 = min(v.shape[3], pretrained_v.shape[3])
                v[:max_0, :max_1, :max_2, :max_3] = pretrained_v[:max_0, :max_1, :max_2, :max_3]
            else:
                raise ValueError("Variable dims > 4 non géré !")
            new_weights.append(v)
    
    target_model.set_weights(new_weights)
    print("Weights transfered successfully !")
        