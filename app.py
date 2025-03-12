from ultralytics import YOLO
import cv2
import streamlit as st
import yaml
import os

def load_yaml(yaml_file):
    with open(yaml_file, 'r') as f:
        return yaml.safe_load(f)

def detect_wrinkles(image_bytes):
    # YOLO modelini yükle
    #model = YOLO('son.pt')
    #model= YOLO('yolv8best.pt')
    model= YOLO('best.pt')
    # YAML dosyasından sınıf isimlerini al
    yaml_data = load_yaml('data.yaml')
    class_names = yaml_data['names']
    
    # Görüntüyü oku
    import numpy as np
    nparr = np.frombuffer(image_bytes.read(), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Tahmin yap
    results = model.predict(source=img, save=False)
    
    # Tespit edilen her sınıf için sayaç ve güven skorları
    class_counts = {}
    class_confidences = {}
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Koordinatları al
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Güven skorunu al
            conf = float(box.conf[0])
            
            # Sınıf indeksini al
            cls = int(box.cls[0])
            class_name = class_names[cls]
            
            # Sınıf sayacını ve güven skorunu güncelle
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            if class_name in class_confidences:
                class_confidences[class_name].append(conf)
            else:
                class_confidences[class_name] = [conf]
            
            # Dikdörtgen ve etiket çiz
            color = (0, 255, 0)  # Yeşil
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f'{class_name}: {conf:.2f}', (x1, y1 - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Ortalama güven skorlarını hesapla
    class_avg_conf = {}
    for class_name, confs in class_confidences.items():
        class_avg_conf[class_name] = sum(confs) / len(confs)
    
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), class_counts, class_avg_conf

def main():
    st.title("Yüz Kırışıklığı Tespit Sistemi")
    
    # Dosya yükleme widget'ı
    uploaded_file = st.file_uploader("Bir fotoğraf seçin...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Orijinal görüntüyü göster
        st.image(uploaded_file, caption='Yüklenen Fotoğraf', use_column_width=True)
        
        # Tespit Et butonu
        if st.button('Tespit Et'):
            with st.spinner('Tespit yapılıyor...'):
                # Görüntüyü işle ve sonucu göster
                result_image, class_counts, class_avg_conf = detect_wrinkles(uploaded_file)
                st.image(result_image, caption='Tespit Sonucu', use_column_width=True)
                
                # Tespit edilen sınıfların sayısını ve güven oranını göster
                if class_counts:
                    st.write("Tespit Edilen Kırışıklıklar:")
                    for class_name, count in class_counts.items():
                        confidence = class_avg_conf[class_name] * 100  # Yüzdeye çevir
                        st.write(f"- {class_name}: {count} adet (Güven Oranı: %{confidence:.1f})")
                else:
                    st.write("Hiç kırışıklık tespit edilemedi.")

if __name__ == "__main__":
    main()
