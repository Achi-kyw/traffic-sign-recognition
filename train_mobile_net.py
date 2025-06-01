import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

def train_model(index, save_dir):
    img_size = (160, 160)
    batch_size = 8
    epochs = 50
    train_dir = 'SignSet/train/' + str(index)
    val_dir = 'SignSet/val/' + str(index)
    model_output_path = f'{save_dir}/model_{str(index)}.h5'

    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        horizontal_flip=False
    )

    train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='sparse'
    )

    val_generator = datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='sparse',
        shuffle=False
    )

    num_classes = train_generator.num_classes

    base_model = MobileNetV2(input_shape=img_size + (3,), include_top=False, weights='imagenet')
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs
    )

    model.save(model_output_path)

    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title(f'Loss and Accuracy for Model_{index}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_plot_{index}.png')

    # 畫 Confusion Matrix
    val_preds = model.predict(val_generator)
    y_pred = np.argmax(val_preds, axis=1)
    y_true = val_generator.classes

    cm = confusion_matrix(y_true, y_pred, normalize='true').T
    labels = list(val_generator.class_indices.keys())

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel='True',
        ylabel='Predicted',
        title=f'Validation Confusion Matrix for Model_{index}'
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/confusion_matrix_{index}.png", dpi=300)

    print(f"Model {index} trained and saved to {model_output_path}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="mobile_net")
    args = parser.parse_args()
    for index in range(6):
        train_model(index, save_dir)

