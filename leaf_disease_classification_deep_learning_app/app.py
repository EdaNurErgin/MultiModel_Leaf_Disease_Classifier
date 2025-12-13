import os
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import joblib

# ===================== DOSYA ADLARI ===================== #
VGG_MODEL_PATH = "vgg16_best_model.h5"
CNN_WEIGHTS_PATH = "best_cnn_model.weights.h5"
SVM_MODEL_PATH = "best_svm_model.joblib"
DENSENET_WEIGHTS_PATH = "denseNet121_final_model.weights.h5"
EFFICIENTNET_WEIGHTS_PATH = "efficientNetB4_final_model.weights.h5"
RESNET_WEIGHTS_PATH = "resNet50_final_model.weights.h5"

# ===================== SINIFLAR ===================== #
CLASS_NAMES = ["Healthy", "Mosaic", "RedRot", "Rust", "Yellow"]
NUM_CLASSES = len(CLASS_NAMES)

# ===================== INPUT BOYUTLARI ===================== #
IMG_SIZE_VGG = (224, 224)
IMG_SIZE_CNN = (224, 224)

IMG_SIZE_DENSENET = (224, 224)  
IMG_SIZE_EFF = (380, 380)       
IMG_SIZE_RESNET = (224, 224)    


# ----------------- yardımcılar ----------------- #
def require_file(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dosya bulunamadı: {path} (app.py ile aynı klasörde olmalı)")


def pil_to_array_0_255(image: Image.Image, size) -> np.ndarray:
    image = image.convert("RGB").resize(size)
    x = np.array(image).astype("float32")  
    return np.expand_dims(x, axis=0)


def show_prediction_result(preds: np.ndarray, model_name: str):
    preds = np.array(preds).flatten()
    idx = int(np.argmax(preds))
    st.success(f"[{model_name}] Tahmin: *{CLASS_NAMES[idx]}*")
    st.write(f"Güven: *{preds[idx]*100:.2f}%*")
    st.subheader("Sınıf olasılıkları")
    for i, name in enumerate(CLASS_NAMES):
        st.write(f"- {name}: {preds[i]*100:.2f}%")


# ===================== 1) VGG16 (TAM MODEL) ===================== #
@st.cache_resource
def load_vgg_model():
    require_file(VGG_MODEL_PATH)
    # compile=False -> sadece inference
    return tf.keras.models.load_model(VGG_MODEL_PATH, compile=False)


def preprocess_vgg(image: Image.Image) -> np.ndarray:
    return pil_to_array_0_255(image, IMG_SIZE_VGG)


# ===================== 2) CUSTOM CNN (WEIGHTS) ===================== #
def build_custom_cnn(input_shape=(224, 224, 3), num_classes=5, dropout_rate=0.5):
    from tensorflow.keras import layers, models
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Rescaling(1.0 / 255),

        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation="softmax"),
    ])
    return model


@st.cache_resource
def load_custom_cnn():
    require_file(CNN_WEIGHTS_PATH)
    model = build_custom_cnn(input_shape=IMG_SIZE_CNN + (3,), num_classes=NUM_CLASSES)
    model.load_weights(CNN_WEIGHTS_PATH)
    return model


def preprocess_cnn(image: Image.Image) -> np.ndarray:
    # model içinde Rescaling var
    return pil_to_array_0_255(image, IMG_SIZE_CNN)


# ===================== 3) SVM (JOBLIB) ===================== #
@st.cache_resource
def load_svm_model():
    """Joblib ile kaydedilmiş en iyi SVM modelini yükler."""
    model = joblib.load(SVM_MODEL_PATH)
    return model


@st.cache_resource
def load_feature_extractor():
    """
    SVM için kullanılacak feature extractor modeli.
    """
    from tensorflow.keras import Model

    
    cnn_model = load_custom_cnn()  
    
    feature_extractor = Model(
        inputs=cnn_model.inputs[0],
        outputs=cnn_model.layers[-3].output,  # Dense(256) katmanının çıktısı
    )
    return feature_extractor


def extract_features_for_svm(image: Image.Image) -> np.ndarray:
    """
    Eğitimde yapıldığı gibi feature çıkarır.
    """
    feature_extractor = load_feature_extractor()
    x = preprocess_cnn(image)      # (1, 224, 224, 3)
    feats = feature_extractor.predict(x)      # (1, feature_dim)
    return feats



# ===================== 4) DenseNet121 (WEIGHTS) ===================== #
def build_densenet121_model():
    from tensorflow.keras import layers
    base = tf.keras.applications.DenseNet121(
        include_top=False,
        weights="imagenet",
        input_shape=IMG_SIZE_DENSENET + (3,)
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=IMG_SIZE_DENSENET + (3,))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs)


@st.cache_resource
def load_densenet():
    require_file(DENSENET_WEIGHTS_PATH)
    model = build_densenet121_model()
    model.load_weights(DENSENET_WEIGHTS_PATH)
    return model


def preprocess_densenet(image: Image.Image) -> np.ndarray:
    from tensorflow.keras.applications.densenet import preprocess_input
    x = pil_to_array_0_255(image, IMG_SIZE_DENSENET)
    return preprocess_input(x)


# ===================== 5) EfficientNetB4 (WEIGHTS) ===================== #
def build_efficientnetb4_model():
    from tensorflow.keras import layers
    base = tf.keras.applications.EfficientNetB4(
        include_top=False,
        weights="imagenet",
        input_shape=IMG_SIZE_EFF + (3,)
    )

   
    base.trainable = True
    fine_tune_at = len(base.layers) - 120
    for i, layer in enumerate(base.layers):
        layer.trainable = (i >= fine_tune_at)

    inputs = tf.keras.Input(shape=IMG_SIZE_EFF + (3,))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs)


@st.cache_resource
def load_efficientnet():
    require_file(EFFICIENTNET_WEIGHTS_PATH)
    model = build_efficientnetb4_model()
    model.load_weights(EFFICIENTNET_WEIGHTS_PATH)
    return model


def preprocess_efficientnet(image: Image.Image) -> np.ndarray:
    from tensorflow.keras.applications.efficientnet import preprocess_input
    x = pil_to_array_0_255(image, IMG_SIZE_EFF)  # 380x380 !!!
    return preprocess_input(x)


# ===================== 6) ResNet50 (WEIGHTS) ===================== #
def build_resnet50_model():
    from tensorflow.keras import layers
    base = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=IMG_SIZE_RESNET + (3,)
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=IMG_SIZE_RESNET + (3,))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs)


@st.cache_resource
def load_resnet():
    require_file(RESNET_WEIGHTS_PATH)
    model = build_resnet50_model()
    model.load_weights(RESNET_WEIGHTS_PATH)
    return model


def preprocess_resnet(image: Image.Image) -> np.ndarray:
    from tensorflow.keras.applications.resnet import preprocess_input
    x = pil_to_array_0_255(image, IMG_SIZE_RESNET)
    return preprocess_input(x)


# ===================== STREAMLIT UI ===================== #
def main():
    st.title("🌿 Yaprak Hastalığı Sınıflandırma Uygulaması")

    model_choice = st.selectbox(
        "Model seç:",
        (
            "VGG16 (h5)",
            "Custom CNN (weights)",
            "SVM (joblib)",
            "DenseNet121 (weights)",
            "EfficientNetB4 (weights)",
            "ResNet50 (weights)",
        )
    )

    uploaded_file = st.file_uploader("Yaprak görseli yükle (JPG/PNG):", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Yüklenen Görsel", use_container_width=True)

        if st.button("Tahmin Et"):
            if model_choice == "VGG16 (h5)":
                model = load_vgg_model()
                x = preprocess_vgg(image)
                preds = model.predict(x, verbose=0)[0]
                show_prediction_result(preds, "VGG16")

            elif model_choice == "Custom CNN (weights)":
                model = load_custom_cnn()
                x = preprocess_cnn(image)
                preds = model.predict(x, verbose=0)[0]
                show_prediction_result(preds, "Custom CNN")

                
            elif model_choice == "SVM (joblib)":
                with st.spinner("SVM modeli tahmin yapıyor..."):
                    svm = load_svm_model()
                    feats = extract_features_for_svm(image)   # (1, feature_dim)
                    pred_idx = int(svm.predict(feats)[0])

                    # SVM etiketi index ise CLASS_NAMES üzerinden map ediyoruz
                    if isinstance(pred_idx, (int, np.integer)):
                        predicted_label = CLASS_NAMES[pred_idx]
                    else:
                        predicted_label = str(pred_idx)

                    st.success(f"[SVM] Tahmin: **{predicted_label}**")
                    st.write(
                        "Not: SVM için olasılık yerine direkt sınıf tahmini gösterilmektedir."
                    )


            elif model_choice == "DenseNet121 (weights)":
                model = load_densenet()
                x = preprocess_densenet(image)
                preds = model.predict(x, verbose=0)[0]
                show_prediction_result(preds, "DenseNet121")

            elif model_choice == "EfficientNetB4 (weights)":
                model = load_efficientnet()
                x = preprocess_efficientnet(image)
                preds = model.predict(x, verbose=0)[0]
                show_prediction_result(preds, "EfficientNetB4")

            else:  # ResNet50
                model = load_resnet()
                x = preprocess_resnet(image)
                preds = model.predict(x, verbose=0)[0]
                show_prediction_result(preds, "ResNet50")


if __name__ == "__main__":
    main()
