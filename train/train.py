# train/train.py
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import json
from model.model import CNNModel  # Import the CNNModel class

class Trainer:
    def __init__(self, base_dir, num_classes):
        """
        Initializes the Trainer class.
        Args:
        - base_dir: The base directory containing participant data.
        - num_classes: Number of classes in the dataset (3 or 7).
        """
        self.base_dir = base_dir
        self.num_classes = num_classes

    def plot_training_history(self, history, participant_folder):
        """ Plot and save training/validation loss and accuracy """
        plt.figure()
        plt.plot(history.history['loss'], label='train loss')
        plt.plot(history.history['val_loss'], label='val loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(participant_folder, 'loss_plot.png'))

        plt.figure()
        plt.plot(history.history['accuracy'], label='train accuracy')
        plt.plot(history.history['val_accuracy'], label='val accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(os.path.join(participant_folder, 'accuracy_plot.png'))

    def save_classification_report(self, y_true, y_pred, target_names, participant_folder):
        """ Save classification report to a txt file """
        report = classification_report(y_true, y_pred, target_names=target_names, digits=4)
        with open(os.path.join(participant_folder, 'classification_report.txt'), 'w') as f:
            f.write(report)

    def save_confusion_matrix(self, y_true, y_pred, target_names, participant_folder):
        """ Save confusion matrix as an image """
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
        disp = disp.plot(cmap=plt.cm.Blues, xticks_rotation='45')
        plt.savefig(os.path.join(participant_folder, 'confusion_matrix.png'))

    def train_model_for_participant(self, participant):
        """
        Train the model for a given participant and save the results.
        Args:
        - participant: The participant's folder name.
        """
        participant_folder = os.path.join(self.base_dir, participant)
        results_folder = os.path.join(participant_folder, 'results')
        os.makedirs(results_folder, exist_ok=True)

        # Data generators
        train_datagen = ImageDataGenerator(rescale=1.0 / 255.0, width_shift_range=0.1, brightness_range=[0.2, 1.0],
                                           height_shift_range=0.1, horizontal_flip=True, rotation_range=45)
        test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

        train_it = train_datagen.flow_from_directory(os.path.join(participant_folder, 'train'),
                                                     class_mode='categorical', batch_size=16, target_size=(224, 224))
        val_it = test_datagen.flow_from_directory(os.path.join(participant_folder, 'val'),
                                                  class_mode='categorical', batch_size=16, target_size=(224, 224))

        # Create and compile the model
        cnn_model = CNNModel(input_shape=(224, 224, 3), num_classes=self.num_classes)
        model = cnn_model.create_model()

        # Callbacks
        es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=10)
        mc = ModelCheckpoint(os.path.join(results_folder, 'best_model.h5'), monitor='val_accuracy', mode='max',
                             verbose=1, save_best_only=True)

        # Train the model
        history = model.fit(train_it, steps_per_epoch=len(train_it),
                            validation_data=val_it, validation_steps=len(val_it),
                            epochs=50, callbacks=[es, mc], verbose=1)

        # Plot results
        self.plot_training_history(history, results_folder)

        # Save training history
        with open(os.path.join(results_folder, 'training_history.json'), 'w') as f:
            json.dump(history.history, f)

        # Evaluate best model
        best_model = load_model(os.path.join(results_folder, 'best_model.h5'))
        val_acc = best_model.evaluate(val_it, steps=len(val_it), verbose=1)[1]
        print(f'Validation accuracy for {participant}: {val_acc * 100.0:.2f}%')

        # Evaluate on the test set
        test_it = test_datagen.flow_from_directory(os.path.join(participant_folder, 'test'),
                                                   class_mode='categorical', batch_size=32, target_size=(224, 224),
                                                   shuffle=False)
        y_true = test_it.classes
        y_pred = np.argmax(best_model.predict(test_it, steps=len(test_it)), axis=1)

        target_names = ['Cool', 'Neutral', 'Warm'] if self.num_classes == 3 else \
            ['Cold', 'Cool', 'Hot', 'Neutral', 'Slightly Cool', 'Slightly Warm', 'Warm']

        # Save classification report and confusion matrix
        self.save_classification_report(y_true, y_pred, target_names, results_folder)
        self.save_confusion_matrix(y_true, y_pred, target_names, results_folder)
