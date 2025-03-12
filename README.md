# **Puff Eyes Detection with YOLOv8** 🧐💡  

Bu proje, **YOLOv8** kullanılarak **şişkin göz tespiti (Puff Eyes Detection)** yapmak için geliştirilmiştir. **1.500 adet fotoğraf** özenle etiketlenmiş ve **YOLOv8'in katman derinlikleri optimize edilerek** eğitilmiştir.  

---

## **📌 Proje Adımları**  

### **1️⃣ Veri Toplama & Etiketleme**  
- **1.500 adet şişkin göz (puff eyes) içeren fotoğraf** toplandı.  
- Görüntüler, **Roboflow** veya **LabelImg** gibi araçlar kullanılarak **etiketlendi**.  
- **YOLO formatında (txt)** etiketlenen veriler eğitim için hazırlandı.  

---

### **2️⃣ YOLOv8 Yapılandırma & Optimizasyon**  
- **Ultralytics YOLOv8** modeli kullanıldı.  
- Modelin **katman derinliği optimize edilerek** en uygun yapı belirlendi.  
- `yolov8n.yaml`, `yolov8s.yaml` vb. konfigürasyonlar değiştirildi.  
- Veri seti eğitim/validasyon/test olarak **%80 - %10 - %10** oranında ayrıldı.  

---

### **3️⃣ Model Eğitimi**  
- **Google Colab + GPU** ortamında eğitim yapıldı.  
- Aşağıdaki kod ile eğitim başlatıldı:  
  ```python
  from ultralytics import YOLO

  # Modeli yükleme ve eğitime başlama
  model = YOLO("yolov8n.yaml")  # Model konfigürasyonu
  model.train(data="data.yaml", epochs=100, imgsz=640, batch=16)
  ```
- **100 epoch boyunca** model eğitildi.  

---

### **4️⃣ Değerlendirme & Test**  
- Eğitim sonrası model, **F1-Score, mAP (Mean Average Precision), Precision, Recall** gibi metriklerle değerlendirildi.  
- Test verileri üzerinde aşağıdaki gibi tahmin yapıldı:  
  ```python
  results = model("test.jpg", save=True, conf=0.5)
  ```
- Tespit edilen **şişkin gözler kutular içinde gösterildi**.  

---

### **5️⃣ Sonuçlar & Çıktılar**  
- Eğitilen model, **gerçek zamanlı ve statik görüntülerde yüksek doğruluk oranı ile çalışmaktadır**.  
- Tespit edilen nesneler **şişkin gözler (puff eyes)** şeklinde işaretlenmiştir.  
- Aşağıda örnek çıktı gösterilmektedir:  

  ![Puff Eyes Detection](example_output.jpg)  

---

## **🚀 Kullanım**  
1. **Modeli yükleyin:**  
   ```sh
   git clone https://github.com/omergzllr/puff-eyes-detection.git
   cd puff-eyes-detection
   ```
2. **YOLOv8'i yükleyin:**  
   ```sh
   pip install ultralytics
   ```
3. **Modeli test edin:**  
   ```python
   from ultralytics import YOLO

   model = YOLO("best.pt")  # Eğitilmiş model dosyanızı buraya ekleyin
   results = model("test.jpg", save=True, conf=0.5)
   ```

---

## **🛠️ Kullanılan Teknolojiler**  
- **📌 YOLOv8 (Ultralytics)**
- **📌 Python**
- **📌 OpenCV**
- **📌 Roboflow / LabelImg (Etiketleme)**
- **📌 Google Colab (GPU destekli eğitim)**

---

## **📩 İletişim & Katkı Sağlama**  
Projeye katkıda bulunmak isterseniz, **Pull Request** gönderebilir veya **Issue** açabilirsiniz!  
📧 Bana ulaşın: omergzllr@gmail.com  

🔹 **GitHub:** https://github.com/omergzllr  
🔹 **LinkedIn:** www.linkedin.com/in/ömergüzeller

 
