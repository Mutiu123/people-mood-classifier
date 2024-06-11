# main.py
import moodClassifier as mdl
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model

def main():
    # Limit GPU memory growth
    mdl.limit_gpu_memory_growth()

    # Load data (already split into train, validation, and test sets)
    data_batch = mdl.load_data('../dataset')
    train_data, val_data, test_data = mdl.split_data(data_batch)

    # Build the model
    my_model = mdl.build_model()

    # Compile the model
    my_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Set up early stopping and model checkpoint callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('../model/best_model.h5', save_best_only=True)

    # Train the model
    history = my_model.fit(train_data, epochs=100, validation_data=val_data,
                           callbacks=[early_stopping, model_checkpoint])

    # Evaluate the model on the test set
    test_loss, test_accuracy = my_model.evaluate(test_data)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # Performance plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].plot(history.history['loss'], color='red', label='loss')
    axes[0].plot(history.history['val_loss'], color='blue', label='val_loss')
    axes[0].set_title('Loss')
    axes[0].legend(loc='upper left')

    axes[1].plot(history.history['accuracy'], color='black', label='accuracy')
    axes[1].plot(history.history['val_accuracy'], color='orange', label='val_accuracy')
    axes[1].set_title('Accuracy')
    axes[1].legend(loc='upper left')

    #Display the figure for a few seconds (e.g., 3 seconds)
    plt.show(block=False)
    plt.pause(5)  # Pause for 3 seconds
    #Close the figure
    plt.close()

    # Evaluate performance metrics
    acc = tf.keras.metrics.BinaryAccuracy()
    preci = tf.keras.metrics.Precision()
    recal = tf.keras.metrics.Recall()

    for batch in test_data.as_numpy_iterator():
        x, y = batch
        yhat = my_model.predict(x)
        acc.update_state(y, yhat)
        preci.update_state(y, yhat)
        recal.update_state(y, yhat)

    print(f'Accuracy: {acc.result().numpy()}, Precision: {preci.result().numpy()}, Recall: {recal.result().numpy()}')

    # Image prediction
    imag = cv2.imread('happy2.jpeg')
    plt.imshow(cv2.cvtColor(imag, cv2.COLOR_BGR2RGB))
    #Display the figure for a few seconds (e.g., 3 seconds)
    plt.show(block=False)
    plt.pause(5)  # Pause for 3 seconds
    #Close the figure
    plt.close()
    
    reSize = tf.image.resize(imag, (256, 256))
    reSize_rgb = cv2.cvtColor(reSize.numpy().astype(np.uint8), cv2.COLOR_BGR2RGB)
    plt.imshow(reSize_rgb)
    plt.show()


    reSize = tf.image.resize(imag, (256, 256))
    reSize_rgb = cv2.cvtColor(reSize.numpy().astype(np.uint8), cv2.COLOR_BGR2RGB)
    #plt.imshow(reSize_rgb)
    #Display the figure for a few seconds (e.g., 3 seconds)
    plt.show(block=False)
    plt.pause(5)  # Pause for 3 seconds
    #Close the figure
    plt.close()

    yhats = my_model.predict(np.expand_dims(reSize / 255, 0))

    if yhats > 0.5:
        print(f'Predicted class is Sad')
    else:
        print(f'Predicted class is Happy')

if __name__ == '__main__':
    main()
