"""
TODO: docstring
"""
import matplotlib.pyplot as pyplot
import os
import tensorflow

class TransferModel:
    """
    TODO: docstring
    """
    def __init__(self):
        """
        TODO: docstring
        """
        # data preprocessing
        path = 'data\\input\\cats_and_dogs_filtered'
        train_dir = os.path.join(path, 'train')
        validation_dir = os.path.join(path, 'validation')
        batch_size = 32
        image_size = (160, 160)
        train_dataset = tensorflow.keras.utils.image_dataset_from_directory(
            train_dir, shuffle=True, batch_size=batch_size, image_size=image_size)
        validation_dataset = tensorflow.keras.utils.image_dataset_from_directory(
            validation_dir, shuffle=True, batch_size=batch_size, image_size=image_size)
        class_names = train_dataset.class_names
        pyplot.figure(figsize=(10, 10))
        for images, labels in train_dataset.take(1):
            for i in range(9):
                ax = pyplot.subplot(3, 3, i + 1)
                pyplot.imshow(images[i].numpy().astype('uint8'))
                pyplot.title(class_names[labels[i]])
                pyplot.axis('off')
        val_batches = tensorflow.data.experimental.cardinality(validation_dataset)
        test_dataset = validation_dataset.take(val_batches // 5)
        validation_dataset = validation_dataset.skip(val_batches // 5)
        print('Number of validation batches: %d' % tensorflow.data.experimental.cardinality(validation_dataset))
        print('Number of test batches: %d' % tensorflow.data.experimental.cardinality(test_dataset))
        # configure the dataset for performance
        autotune = tensorflow.data.AUTOTUNE
        train_dataset = train_dataset.prefetch(buffer_size=autotune)
        validation_dataset = validation_dataset.prefetch(buffer_size=autotune)
        test_dataset = test_dataset.prefetch(buffer_size=autotune)
        # use data augmentation
        data_augmentation = tensorflow.keras.Sequential([
            tensorflow.keras.layers.RandomFlip('horizontal'),
            tensorflow.keras.layers.RandomRotation(0.2)])
        for image, _ in train_dataset.take(1):
            pyplot.figure(figsize=(10, 10))
            first_image = image[0]
            for i in range(9):
                ax = pyplot.subplot(3, 3, i+1)
                augmented_image = data_augmentation(tensorflow.expand_dims(first_image, 0))
                pyplot.imshow(augmented_image[0] / 255)
                pyplot.axis('off')
        # rescale pixel values
        preprocess_input = tensorflow.keras.applications.mobilenet_v2.preprocess_input
        rescale = tensorflow.keras.layers.Rescaling(1.0/127.5, offset=-1)
        # create the base model from the pre-trained model MobileNet V2
        image_shape = image_size + (3,)
        base_model = tensorflow.keras.applications.MobileNetV2(
            input_shape=image_shape, include_top=False, weights='imagenet')
        image_batch, label_batch = next(iter(train_dataset))
        feature_batch = base_model(image_batch)
        print(feature_batch.shape)
        # feature extraction
        base_model.trainable = False # freeze the convolutional base
        base_model.summary()
        # add a classification head
        global_average_layer = tensorflow.keras.layers.GlobalAveragePooling2D()
        feature_batch_average = global_average_layer(feature_batch)
        print(feature_batch_average.shape)
        prediction_layer = tensorflow.keras.layers.Dense(1)
        prediction_batch = prediction_layer(feature_batch_average)
        print(prediction_batch.shape)
        inputs = tensorflow.keras.Input(shape=(160, 160, 3))
        x = data_augmentation(inputs)
        x = preprocess_input(x)
        x = base_model(x, training=False)
        x = global_average_layer(x)
        x = tensorflow.keras.layers.Dropout(0.2)(x)
        outputs = prediction_layer(x)
        model = tensorflow.keras.Model(inputs, outputs)
        # compile the model
        base_learning_rate = 0.0001
        model.compile(
            optimizer=tensorflow.keras.optimizers.Adam(learning_rate=base_learning_rate),
            loss=tensorflow.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=['accuracy'])
        model.summary()
        len(model.trainable_variables)
        # train the model
        initial_epochs = 10
        loss0, accuracy0 = model.evaluate(validation_dataset)
        print("initial loss: {:.2f}".format(loss0))
        print("initial accuracy: {:.2f}".format(accuracy0))
        history = model.fit(train_dataset, epochs=initial_epochs, validation_data=validation_dataset)
        # learning curves
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        pyplot.figure(figsize=(8, 8))
        pyplot.subplot(2, 1, 1)
        pyplot.plot(acc, label='Training Accuracy')
        pyplot.plot(val_acc, label='Validation Accuracy')
        pyplot.legend(loc='lower right')
        pyplot.ylabel('Accuracy')
        pyplot.ylim([min(pyplot.ylim()), 1])
        pyplot.title('Training and Validation Accuracy')
        pyplot.subplot(2, 1, 2)
        pyplot.plot(loss, label='Training Loss')
        pyplot.plot(val_loss, label='Validation Loss')
        pyplot.legend(loc='upper right')
        pyplot.ylabel('Cross Entropy')
        pyplot.ylim([0, 1.0])
        pyplot.title('Training and Validation Loss')
        pyplot.xlabel('epoch')
        pyplot.show()
        base_model.trainable = True # unfreeze the top layers of the model
        # let's take a look to see how many layers are in the base model
        print("Number of layers in the base model:", len(base_model.layers))
        # fine-tune from this layer onwards
        fine_tune_at = 100
        # freeze all the layers before the `fine_tune_at` layer
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        # compile the model
        model.compile(
            loss=tensorflow.keras.losses.BinaryCrossentropy(from_logits=True),
            optimizer=tensorflow.keras.optimizers.RMSprop(learning_rate=base_learning_rate/10),
            metrics=['accuracy'])
        model.summary()
        print(len(model.trainable_variables))
        # continue training the model
        fine_tune_epochs = 10
        total_epochs = initial_epochs + fine_tune_epochs
        history_fine = model.fit(
            train_dataset, epochs=total_epochs,
            initial_epoch=history.epoch[-1],
            validation_data=validation_dataset)
        acc += history_fine.history['accuracy']
        val_acc += history_fine.history['val_accuracy']
        loss += history_fine.history['loss']
        val_loss += history_fine.history['val_loss']
        pyplot.figure(figsize=(8, 8))
        pyplot.subplot(2, 1, 1)
        pyplot.plot(acc, label='Training Accuracy')
        pyplot.plot(val_acc, label='Validation Accuracy')
        pyplot.ylim([0.8, 1])
        pyplot.plot(
            [initial_epochs-1, initial_epochs-1],
            pyplot.ylim(), label='Start Fine Tuning')
        pyplot.legend(loc='lower right')
        pyplot.title('Training and Validation Accuracy')
        pyplot.subplot(2, 1, 2)
        pyplot.plot(loss, label='Training Loss')
        pyplot.plot(val_loss, label='Validation Loss')
        pyplot.ylim([0, 1.0])
        pyplot.plot(
            [initial_epochs-1, initial_epochs-1],
            pyplot.ylim(), label='Start Fine Tuning')
        pyplot.legend(loc='upper right')
        pyplot.title('Training and Validation Loss')
        pyplot.xlabel('epoch')
        pyplot.show()
        # evaluation and prediction
        loss, accuracy = model.evaluate(test_dataset)
        print('Test accuracy:', accuracy)
        # Retrieve a batch of images from the test set
        image_batch, label_batch = test_dataset.as_numpy_iterator().next()
        predictions = model.predict_on_batch(image_batch).flatten()
        # Apply a sigmoid since our model returns logits
        predictions = tensorflow.nn.sigmoid(predictions)
        predictions = tensorflow.where(predictions < 0.5, 0, 1)
        print('Predictions:\n', predictions.numpy())
        print('Labels:\n', label_batch)
        pyplot.figure(figsize=(10, 10))
        for i in range(9):
            ax = pyplot.subplot(3, 3, i + 1)
            pyplot.imshow(image_batch[i].astype('uint8'))
            pyplot.title(class_names[predictions[i]])
            pyplot.axis('off')

if __name__ == '__main__':
    TransferModel()
