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

def create_model(input_shape=(150, 150, 3)):
    """
    Create a CNN model for binary classification
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        Flatten(),
        Dropout(0.5),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model():
    """
    Main training function
    """
    # Set random seed for reproducibility
    random.seed(42)
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Paths
    original_data_dir = '/home/ryokuryu/splat_2/private/start_detect/data'
    balanced_data_dir = '/home/ryokuryu/splat_2/private/start_detect/balanced_data'
    
    # Balance dataset
    balance_dataset(original_data_dir, balanced_data_dir)
    
    # Image dimensions
    img_height, img_width = 150, 150
    batch_size = 32
    
    # Data generators with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        validation_split=0.2
    )
    
    # Training data
    train_generator = train_datagen.flow_from_directory(
        balanced_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary',
        subset='training'
    )
    
    # Validation data
    validation_generator = train_datagen.flow_from_directory(
        balanced_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary',
        subset='validation'
    )
    
    print(f"Training samples: {train_generator.samples}")
    print(f"Validation samples: {validation_generator.samples}")
    print(f"Class indices: {train_generator.class_indices}")
    
    # Create model
    model = create_model((img_height, img_width, 3))
    model.summary()
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=0.0001
    )
    
    # Train model
    print("Starting training...")
    history = model.fit(
        train_generator,
        epochs=50,
        validation_data=validation_generator,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Save model
    model.save('/home/ryokuryu/splat_2/private/start_detect/image_classifier_model.h5')
    print("Model saved successfully!")
    
    # Evaluate model
    print("\nEvaluating model...")
    
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
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    # Plot training & validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('/home/ryokuryu/splat_2/private/start_detect/training_history.png')
    plt.show()
    
    return model, history

if __name__ == "__main__":
    print("Starting image classification training...")
    model, history = train_model()
    print("Training completed!")