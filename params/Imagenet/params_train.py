import params as p


def get_params():
    params = {
        p.INPUT_SHAPE: [256, 256, 1],  # Imagenet size
        p.N_EPOCHS: 50,
        p.BATCH_SIZE: 15,
        p.LR: 0.00003,  # Learning rate

        p.N_IMAGES_TRAIN_VAL: 456567,
        p.TRAIN_SIZE: 0.6,

        p.LOSS: 'cross_entropy_weighted',

        p.OUTPUT_PATH: '/imatge/pvidal/work/dlai/mini_batch_test',

        p.MODEL_NAME: 'cnn'

    }

    return params
