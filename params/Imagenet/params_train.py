import params as p


def get_params():
    params = {
        p.INPUT_SHAPE: [256, 256],  # Imagenet size
        p.N_EPOCHS: 50,
        p.BATCH_SIZE: 1,
        p.LR: 0.0005,  # Learning rate

        p.N_IMAGES_TRAIN_VAL: 1000,
        p.TRAIN_SIZE: 0.6,

        p.LOSS: 'cross_entropy_weighted',

        p.OUTPUT_PATH: '/work/pvidal/',

        p.MODEL_NAME: 'cnn'

    }

    return params