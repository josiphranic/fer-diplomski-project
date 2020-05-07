from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, MaxPooling2D, Dropout, UpSampling2D, Input, concatenate
from tensorflow.keras.applications.vgg19 import VGG19


def upsample_conv(filters, kernel_size, strides, padding):
    return Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)


def upsample_simple(filters, kernel_size, strides, padding):
    return UpSampling2D(strides)


def conv2d_block(
        inputs,
        use_batch_norm=True,
        dropout=0.3,
        filters=16,
        kernel_size=(3, 3),
        activation='relu',
        kernel_initializer='he_normal',
        padding='same'):
    c = Conv2D(filters, kernel_size, activation=activation, kernel_initializer=kernel_initializer, padding=padding)(
        inputs)
    if use_batch_norm:
        c = BatchNormalization()(c)
    if dropout > 0.0:
        c = Dropout(dropout)(c)
    c = Conv2D(filters, kernel_size, activation=activation, kernel_initializer=kernel_initializer, padding=padding)(c)
    if use_batch_norm:
        c = BatchNormalization()(c)
    return c


def conv2d_block_single(
        inputs,
        use_batch_norm=True,
        dropout=0.3,
        filters=16,
        kernel_size=(3, 3),
        activation='relu',
        kernel_initializer='he_normal',
        padding='same'):
    c = Conv2D(filters, kernel_size, activation=activation, kernel_initializer=kernel_initializer, padding=padding)(
        inputs)
    if use_batch_norm:
        c = BatchNormalization()(c)
    if dropout > 0.0:
        c = Dropout(dropout)(c)
    return c


def custom_unet(
        input_shape,
        num_classes=1,
        use_batch_norm=True,
        upsample_mode='deconv',  # 'deconv' or 'simple'
        use_dropout_on_upsampling=False,
        dropout=0.3,
        dropout_change_per_layer=0.0,
        filters=16,
        num_layers=4,
        output_activation='sigmoid'):  # 'sigmoid' or 'softmax'

    if upsample_mode == 'deconv':
        upsample = upsample_conv
    else:
        upsample = upsample_simple

    # Build U-Net model
    inputs = Input(input_shape)
    x = inputs

    down_layers = []
    for l in range(num_layers):
        x = conv2d_block(inputs=x, filters=filters, use_batch_norm=use_batch_norm, dropout=dropout)
        down_layers.append(x)
        x = MaxPooling2D((2, 2))(x)
        dropout += dropout_change_per_layer
        filters = filters * 2  # double the number of filters with each layer

    x = conv2d_block(inputs=x, filters=filters, use_batch_norm=use_batch_norm, dropout=dropout)

    if not use_dropout_on_upsampling:
        dropout = 0.0
        dropout_change_per_layer = 0.0

    for conv in reversed(down_layers):
        filters //= 2  # decreasing number of filters with each layer
        dropout -= dropout_change_per_layer
        x = upsample(filters, (2, 2), strides=(2, 2), padding='same')(x)
        x = concatenate([x, conv])
        x = conv2d_block(inputs=x, filters=filters, use_batch_norm=use_batch_norm, dropout=dropout)

    outputs = Conv2D(num_classes, (1, 1), activation=output_activation)(x)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model


def get_frozen_pretrained_vgg19_model(input_shape):
    model = VGG19(include_top=False, input_shape=(input_shape[0], input_shape[1], 3))
    model.trainable = False
    return model


def custom_unet_with_vgg19_encoder(input_shape,
                                   num_classes=1,
                                   use_batch_norm=True,
                                   upsample_mode='deconv',
                                   use_dropout_on_upsampling=False,
                                   dropout=0.3,
                                   output_activation='sigmoid'):
    encoder = get_frozen_pretrained_vgg19_model(input_shape)
    if upsample_mode == 'deconv':
        upsample = upsample_conv
    else:
        upsample = upsample_simple
    if not use_dropout_on_upsampling:
        dropout = 0.0
    down_layer = [layer for layer in encoder.layers if layer.name in ['block1_conv2',
                                                                      'block2_conv2',
                                                                      'block3_conv4',
                                                                      'block4_conv4',
                                                                      'block5_conv4']]
    down_layer = reversed(down_layer)
    x = encoder.layers[-1].output
    x = conv2d_block_single(inputs=x, filters=512, use_batch_norm=use_batch_norm, dropout=dropout)
    for layer in down_layer:
        filters = int(layer.filters / 2)
        x = upsample(filters, (2, 2), strides=(2, 2), padding='same')(x)
        x = concatenate([x, layer.output])
        filters = layer.filters
        x = conv2d_block_single(inputs=x, filters=filters, use_batch_norm=use_batch_norm, dropout=dropout)

    output = Conv2D(num_classes, (1, 1), activation=output_activation)(x)

    return Model(inputs=encoder.inputs, outputs=[output])
