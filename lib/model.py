import tensorflow as tf

_INITIAL_EPOCHS = 20
_FINE_TUNE_EPOCHS = 10


def model_info(model_name):
    _, base_model_name, num_classes = model_name.split('-')
    num_classes = int(num_classes[1:])
    return [base_model_name, num_classes]


def transfer_learn(model, train_dataset, validation_dataset, epochs=_INITIAL_EPOCHS, verbose=0):
    loss0, accuracy0 = model.evaluate(validation_dataset)

    print("initial loss: {:.2f}".format(loss0))
    print("initial accuracy: {:.2f}".format(accuracy0))

    save_name = f'best_{model.name}.h5'
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=verbose, patience=8)
    mc = tf.keras.callbacks.ModelCheckpoint(save_name, monitor='val_accuracy', mode='max', verbose=verbose,
                                            save_best_only=True)

    history = model.fit(train_dataset,
                        epochs=epochs,
                        validation_data=validation_dataset,
                        callbacks=[es, mc])

    return history


def fine_tune(model, train_dataset, validation_dataset, epochs=_FINE_TUNE_EPOCHS, initial_epochs=_INITIAL_EPOCHS,
              from_layer=100, learning_rate=0.00001, verbose=1):
    base_model_name, num_classes = model_info(model.name)
    base_model = model.get_layer(base_model_name)

    base_model.trainable = True

    # Fine-tune from this layer onwards
    fine_tune_at = from_layer

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    loss_fun = tf.keras.losses.BinaryCrossentropy(
        from_logits=True) if num_classes <= 2 else tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(loss=loss_fun,
                  optimizer=tf.keras.optimizers.RMSprop(lr=learning_rate),
                  metrics=['accuracy'])

    save_name = f'best_fine_{model.name}.h5'
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=verbose, patience=8)
    mc = tf.keras.callbacks.ModelCheckpoint(save_name, monitor='val_accuracy', mode='max', verbose=0,
                                            save_best_only=True)

    total_epochs = initial_epochs + epochs

    history = model.fit(train_dataset,
                        epochs=total_epochs,
                        initial_epoch=initial_epochs,
                        validation_data=validation_dataset,
                        callbacks=[es, mc])

    return history
