from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Conv2D


def get_custom_model_with_pretrained_encoder(model_path, num_classes, trainable_encoder):
    model = load_model(model_path, custom_objects={'jaccard_loss': jaccard_loss,
                                                   'dice_coef': dice_coef})
    for layer in model.layers[:int(len(model.layers) / 2) + 1]:
        layer.trainable = trainable_encoder
    output_layer = model.layers[-1]
    new_output_layer = Conv2D(num_classes,
                              output_layer.kernel_size,
                              activation=output_layer.activation,
                              name=output_layer.name)(output_layer.input)
    return Model(inputs=model.inputs, outputs=[new_output_layer])


def jaccard_loss(y_true, y_pred, smooth=100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth


def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def custom_generalized_dice_loss(y_true, y_pred):
    number_of_classes = K.shape(y_true).shape[-1]
    weights = [1. / K.sum(y_true[:, :, :, class_index]) for class_index in range(number_of_classes)]
    weights = [weight / K.sum(weights) for weight in weights]
    dice_coefs = [weights[class_index] * dice_coef(y_true[:, :, :, class_index], y_pred[:, :, :, class_index]) for class_index in range(number_of_classes)]
    return 1. - K.sum(dice_coefs)
