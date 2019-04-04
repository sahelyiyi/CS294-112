from keras import models
from keras.engine.saving import load_model
from keras.layers import Dense, Dropout


def build_model(cfg, input_size, output_size):
    # Build neural network
    model = models.Sequential()
    model.add(Dense(cfg.MODEL.NN.INPUT_LAYER[0]['units'], activation=cfg.MODEL.NN.INPUT_LAYER[0]['activation'], input_shape=(input_size,)))
    for hidden_layer in cfg.MODEL.NN.HIDDEN_LAYERS:
        model.add(Dense(hidden_layer['units'], activation=hidden_layer['activation']))
    model.add(Dense(output_size, activation=cfg.MODEL.NN.OUTPUT_LAYER[0]['activation']))
    model.add(Dropout(cfg.MODEL.NN.DROPOUT))

    # Compile model_data
    model.compile(optimizer=cfg.MODEL.NN.OPTIMIZER,
                  loss=cfg.MODEL.NN.LOSS,
                  metrics=cfg.MODEL.NN.METRICS)
    return model


def train(cfg, model, model_file_path, X_train, y_train):
    model.fit(X_train, y_train,
              batch_size=cfg.INPUT.BATCH_SIZE,
              epochs=cfg.INPUT.EPOCHS,
              # callbacks=[plot_losses],
              verbose=1)
    model.save(model_file_path)


def test(model_file_path, X_test, y_test):
    model = load_model(model_file_path)
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])