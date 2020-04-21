from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Conv2D


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
