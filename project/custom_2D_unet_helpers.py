from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Conv2D
import numpy as np


def get_custom_model_with_frozen_encoder(model_path, num_classes):
    model = load_model(model_path, custom_objects={'jaccard_distance': jaccard_distance})
    for layer in model.layers[:int(len(model.layers) / 2) + 1]:
        layer.trainable = False
    output_layer = model.layers[-1]
    new_output_layer = Conv2D(num_classes,
                              output_layer.kernel_size,
                              activation=output_layer.activation,
                              name=output_layer.name)(output_layer.input)
    return Model(inputs=model.inputs, outputs=[new_output_layer])


def jaccard_distance(y_true, y_pred, smooth=100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def hausdorff_distance(true_multiclass_matrix_mask, predicted_multiclass_matrix_mask):
    # euclidean distance = numpy.linalg.norm(a-b)
    hs = []
    for class_index in range(0, true_multiclass_matrix_mask.shape[2]):
        predicted_flatten = []
        true_flatten = []
        for x, y in np.ndindex(predicted_multiclass_matrix_mask.shape[:2]):
            if predicted_multiclass_matrix_mask[x][y][class_index] is not 0:
                predicted_flatten.append(np.array((x, y)))
            if true_multiclass_matrix_mask[x][y][class_index] is not 0:
                true_flatten.append(np.array((x, y)))
        h = max([min([np.linalg.norm(t_xy-p_xy) for p_xy in predicted_flatten]) for t_xy in true_flatten])
        hs.append(h)
    return hs
