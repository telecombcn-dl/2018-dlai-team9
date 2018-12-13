import params as p


def get_params():
    params = {
        p.INPUT_SHAPE: [256, 256, 1],  # Imagenet size
        p.N_EPOCHS: 50,
        p.BATCH_SIZE: 25,
        p.LR:0.0001,  # Learning rate
        p.N_IMAGES_TRAIN_VAL: 12677,
        p.TRAIN_SIZE: 0.8,

        p.LOSS: 'cross_entropy_weighted',

        p.OUTPUT_PATH: '/work/pvidal/',

        p.MODEL_NAME: 'cnn'

    }

    return params
