import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os

# Sayfa yapılandırması
st.set_page_config(page_title="Beyin MRI Sınıflandırma", layout="centered")

# Başlık
st.title("🧠 Beyin MRI")
st.markdown("""
Bu uygulama, yüklediğiniz MRI görüntüsünü sınıflandırır ve Grad-CAM ile açıklanabilirlik sunar.  
Model dört sınıf arasında tahmin yapar.
""")

# 🔹 Modeli yükle
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mri_classifier_gradcam.keras", compile=False)

model = load_model()
class_names = ["0.MildDemented", "1.ModerateDemented", "2.NonDemented", "3.VeryMildDemented"]

# 🔹 Grad-CAM hesaplama fonksiyonu
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="last_conv", pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        tape.watch(conv_outputs)

        if isinstance(predictions, list):
            predictions = predictions[0]

        # 🔧 predictions şekline göre argmax
        if pred_index is None:
            if len(predictions.shape) == 2:
                pred_index = int(tf.argmax(predictions[0]).numpy())
            elif len(predictions.shape) == 1:
                pred_index = int(tf.argmax(predictions).numpy())
            else:
                raise ValueError(f"Beklenmeyen predictions şekli: {predictions.shape}")

        class_channel = predictions[0][pred_index] if len(predictions.shape) == 2 else predictions[pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# 🔹 Görselleştirme fonksiyonu
def overlay_gradcam_on_image(gray_img, heatmap, alpha=0.4):
    rgb_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
    heatmap_resized = cv2.resize(heatmap, (rgb_img.shape[1], rgb_img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(rgb_img, 1 - alpha, heatmap_color, alpha, 0)
    return superimposed_img

# 🔹 Görüntü yükleme
uploaded_file = st.file_uploader(
    "📤 Lütfen bir beyin MRI görüntüsü yükleyin (JPG veya PNG formatında)",
    type=["jpg", "png"],
    help="Görüntü 128x128 boyutuna yeniden boyutlandırılacaktır (gri tonlamalı)."
)

use_sample = False
if uploaded_file is None:
    st.info("💡 Henüz bir görüntü yüklemediniz. Aşağıdan örnek görüntü kullanabilirsiniz.")
    if st.button("🖼️ Örnek MRI Görüntüsünü Kullan"):
        use_sample = True
        sample_path = "sample_mri.jpg"
        if not os.path.exists(sample_path):
            st.error("Örnek görüntü bulunamadı. Lütfen 'sample_mri.jpg' dosyasını ekleyin.")
            st.stop()
        img = cv2.imread(sample_path)
else:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

# 🔹 Görüntü varsa işle
if uploaded_file or use_sample:
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (128, 128))
    st.image(img_gray, caption="🧠 Yüklenen MRI Görüntüsü (Gri Tonlamalı)", use_container_width=True)

    # 🔹 Model tahmini
    input_tensor = np.expand_dims(img_resized / 255.0, axis=(0, -1))  # (1, 128, 128, 1)
    prediction = model.predict(input_tensor)[0]
    predicted_class = class_names[np.argmax(prediction)]

    st.subheader(f"🔍 Tahmin Edilen Sınıf: **{predicted_class}**")
    st.markdown("📊 Sınıf olasılıkları:")
    st.bar_chart(dict(zip(class_names, prediction)))

    # 🔹 Grad-CAM hesapla ve görselleştir
    heatmap = make_gradcam_heatmap(input_tensor, model, last_conv_layer_name="last_conv")
    cam = overlay_gradcam_on_image(img_resized, heatmap)

    st.image(cam, caption="🔥 Grad-CAM Görselleştirmesi", use_container_width=True)
    st.markdown("""
    🔍 Bu görselleştirme, modelin tahmin yaparken odaklandığı bölgeleri gösterir.  
    Kırmızı alanlar, karar üzerinde en etkili bölgeleri temsil eder.""")