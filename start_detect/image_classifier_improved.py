import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import random
import shutil
import matplotlib.pyplot as plt
from PIL import Image

def balance_dataset(source_dir, target_dir, min_samples=139):
    """
    Balance the dataset by taking equal number of samples from each class
    """
    print(f"Balancing dataset with {min_samples} samples per class...")
    
    # Create balanced dataset directory
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)
    
    # Create subdirectories for each class
    for class_name in ['not', 'start']:
        os.makedirs(os.path.join(target_dir, class_name))
    
    # Balance each class
    for class_name in ['not', 'start']:
        source_class_dir = os.path.join(source_dir, class_name)
        target_class_dir = os.path.join(target_dir, class_name)
        
        # Get all image files
        image_files = [f for f in os.listdir(source_class_dir) if f.endswith('.png')]
        
        # Randomly sample min_samples images
        selected_images = random.sample(image_files, min_samples)
        
        # Copy selected images to balanced dataset
        for img_file in selected_images:
            shutil.copy2(
                os.path.join(source_class_dir, img_file),
                os.path.join(target_class_dir, img_file)
            )
        
        print(f"Copied {len(selected_images)} images for class '{class_name}'")
    
    print("Dataset balanced successfully!")

def create_simple_model(input_shape=(64, 64, 1)):
    """
    Create a simpler CNN model for grayscale binary classification
    """
    model = Sequential([
        # First conv block
        Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Second conv block
        Conv2D(32, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Third conv block
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Dense layers
        Flatten(),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.0005),  # Lower learning rate
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

class GrayscaleImageDataGenerator(ImageDataGenerator):
    """
    Custom ImageDataGenerator that converts images to grayscale
    """
    def flow_from_directory(self, *args, **kwargs):
        kwargs['color_mode'] = 'grayscale'  # Force grayscale
        return super().flow_from_directory(*args, **kwargs)

def train_improved_model():
    """
    Main training function with improvements
    """
    # Set random seed for reproducibility
    random.seed(42)
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Paths
    original_data_dir = '/home/ryokuryu/splat_2/private/start_detect/data'
    balanced_data_dir = '/home/ryokuryu/splat_2/private/start_detect/balanced_data_improved'
    
    # Balance dataset
    balance_dataset(original_data_dir, balanced_data_dir)
    
    # Image dimensions - smaller for simpler images
    img_height, img_width = 64, 64
    batch_size = 16  # Smaller batch size
    
    # Data generators with stronger augmentation for grayscale
    train_datagen = GrayscaleImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        horizontal_flip=True,
        zoom_range=0.3,
        shear_range=0.2,
        brightness_range=[0.8, 1.2],
        validation_split=0.2
    )
    
    # Training data
    train_generator = train_datagen.flow_from_directory(
        balanced_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary',
        subset='training',
        color_mode='grayscale'
    )
    
    # Validation data
    validation_generator = train_datagen.flow_from_directory(
        balanced_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary',
        subset='validation',
        color_mode='grayscale'
    )
    
    print(f"Training samples: {train_generator.samples}")
    print(f"Validation samples: {validation_generator.samples}")
    print(f"Class indices: {train_generator.class_indices}")
    
    # Create simpler model
    model = create_simple_model((img_height, img_width, 1))
    model.summary()
    
    # Callbacks with more aggressive early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,  # More patience
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,  # More aggressive reduction
        patience=7,
        min_lr=0.00001,
        verbose=1
    )
    
    # Train model
    print("Starting improved training...")
    history = model.fit(
        train_generator,
        epochs=100,  # More epochs with early stopping
        validation_data=validation_generator,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Save model
    model.save('/home/ryokuryu/splat_2/private/start_detect/image_classifier_improved_model.h5')
    print("Improved model saved successfully!")
    
    # Evaluate model
    print("\nEvaluating improved model...")
    
    # Test on validation data
    val_loss, val_accuracy = model.evaluate(validation_generator, verbose=0)
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")
    
    # Get predictions for classification report
    validation_generator.reset()
    predictions = model.predict(validation_generator, verbose=0)
    predicted_classes = (predictions > 0.5).astype(int).flatten()
    true_classes = validation_generator.classes
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(true_classes, predicted_classes, 
                              target_names=['not', 'start']))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(true_classes, predicted_classes)
    print(cm)
    
    # Calculate and display additional metrics
    tn, fp, fn, tp = cm.ravel()
    precision_not = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_not = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision_start = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_start = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"\nDetailed Metrics:")
    print(f"True Negatives: {tn}, False Positives: {fp}")
    print(f"False Negatives: {fn}, True Positives: {tp}")
    print(f"Precision for 'not': {precision_not:.4f}")
    print(f"Recall for 'not': {recall_not:.4f}")
    print(f"Precision for 'start': {precision_start:.4f}")
    print(f"Recall for 'start': {recall_start:.4f}")
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    # Plot training & validation accuracy
    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy (Improved)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot training & validation loss
    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss (Improved)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot learning rate
    plt.subplot(1, 3, 3)
    plt.plot(history.history['learning_rate'], label='Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('/home/ryokuryu/splat_2/private/start_detect/training_history_improved.png', dpi=300)
    plt.show()
    
    # Analysis of overfitting
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    overfitting_gap = final_train_acc - final_val_acc
    
    print(f"\nOverfitting Analysis:")
    print(f"Final Training Accuracy: {final_train_acc:.4f}")
    print(f"Final Validation Accuracy: {final_val_acc:.4f}")
    print(f"Overfitting Gap: {overfitting_gap:.4f}")
    
    if overfitting_gap > 0.1:
        print("WARNING: Model is still overfitting (gap > 0.1)")
    elif overfitting_gap > 0.05:
        print("CAUTION: Model shows some overfitting (gap > 0.05)")
    else:
        print("GOOD: Model shows minimal overfitting")
    
    return model, history

if __name__ == "__main__":
    print("Starting improved image classification training...")
    print("Key improvements:")
    print("- Grayscale conversion (1 channel instead of 3)")
    print("- Smaller image size (64x64 instead of 150x150)")
    print("- Simpler model architecture")
    print("- Stronger regularization")
    print("- Enhanced data augmentation")
    print("- Lower learning rate")
    print("- Smaller batch size")
    print()
    
    model, history = train_improved_model()
    print("Improved training completed!")