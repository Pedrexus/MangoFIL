from mango.models import get_model
import mango

def create_io():
    root = 'MangoFIL/data'
    exclude_dirs = ('FIL_NANDA', 'EXCLUDED', 'NPZ_FILES')

    io = mango.IO(root, exclude_dirs)

    return io

def test_training_model():
    io = create_io()
    x, y = io.load(resize=.1, parallel=True)

    model = get_model('LeNet5') # leNetFixed

    from tensorflow.keras.optimizers import Adam
    trainer = mango.Trainer(
        x, y, model, test_size = .25, validation_size = .15,
    ) # normalize data

    result = trainer.train(
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', 'binary_f1_score'], 
        optimizer=Adam(lr=1e-3),
        shuffle=True, epochs=1, verbose=2, batch_size=256, random_state=123,
        augmentation=dict(
            rotation_range=20, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2, 
            shear_range=0.15, horizontal_flip=True, fill_mode="nearest"
        )
    )

    from numpy import argmax
    best_epoch = argmax(abs(result.history['val_f1_score']))

    assert best_epoch == 1