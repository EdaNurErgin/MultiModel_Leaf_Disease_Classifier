🌿 Multi-Model Leaf Disease Classifier

Bu proje, yaprak görüntülerinden hastalık tespiti yapmak için geliştirilmiş bir görüntü sınıflandırma uygulamasıdır.
Amaç, farklı makine öğrenmesi ve derin öğrenme modellerini kullanarak aynı veri üzerinde karşılaştırmalı sonuçlar elde etmektir.

Kullanıcı bir yaprak fotoğrafı yükler, seçtiği model görüntüyü analiz eder ve tahmin edilen hastalık türünü gösterir.
Uygulama Streamlit ile hazırlanmış basit bir web arayüzüne sahiptir.


🔍 Neler Yapıyor?

Yaprak fotoğrafı alır
Seçilen modeli kullanarak tahmin yapar
Tahmin edilen sınıfı ve güven oranını gösterir
Farklı modeller arasında karşılaştırma yapma imkânı sunar


🧠 Kullanılan Modeller

Bu projede toplam 6 farklı model bulunmaktadır:
VGG16
Custom CNN (sıfırdan oluşturulmuş)
SVM (CNN’den çıkarılan özellikler ile)
DenseNet121
EfficientNetB4
ResNet50
Tüm modeller önceden eğitilmiştir ve uygulama sırasında tekrar eğitilmez.


🧪 Tahmin Edilen Sınıflar
Modeller aşağıdaki sınıflardan birini tahmin eder:
Healthy
Mosaic
RedRot
Rust
Yellow


🖥️ Uygulama Arayüzü

Model seçimi yapılabilir
Görsel yüklenir
Tek tıkla tahmin alınır
Sonuç ekranda gösterilir
Arayüz karmaşık değildir, özellikle eğitim amaçlı hazırlanmıştır.


📁 Klasör Yapısı
MultiModel_Leaf_Disease_Classifier/
│
├── leaf_disease_classification_deep_learning_app/
│   ├── app.py
│   ├── model dosyaları (.h5, .weights.h5, .joblib)
│
├── notebooks/
│   ├── veri hazırlama
│   ├── model eğitim notebook’ları
│
├── models and reports/
│   ├── her model için sonuçlar ve grafikler
│
├── .gitignore
├── .gitattributes
└── README.md


⚙️ Nasıl Çalıştırılır?

1. Repoyu klonla
git clone https://github.com/cemilenurerden/MultiModel_Leaf_Disease_Classifier.git
cd MultiModel_Leaf_Disease_Classifier

2. Model dosyalarını çek
git lfs install
git lfs pull

3. Sanal ortam oluştur
python -m venv venv
venv\Scripts\activate

4. Gerekli paketleri yükle
pip install -r requirements.txt

5. Uygulamayı başlat
streamlit run leaf_disease_classification_deep_learning_app/app.py

📦 Model Dosyaları Hakkında

Model dosyaları büyük olduğu için Git LFS kullanılmıştır.
Repo’yu indirdikten sonra modeller görünmüyorsa:

git lfs pull komutunu çalıştırman gerekir.


📊 Sonuçlar

Her model için:
Doğruluk ve kayıp grafikleri
Confusion matrix
ROC eğrileri
Performans raporları
models and reports klasörü altında bulunmaktadır.


🛠️ Kullanılan Teknolojiler

Python
TensorFlow / Keras
Scikit-learn
Streamlit
Git & Git LFS
