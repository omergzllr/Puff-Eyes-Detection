from ultralytics import YOLO
import torch
import torch.nn as nn
from ultralytics.nn.modules import C2f, SPPF, Conv

class CustomYOLOv8(YOLO):
    def __init__(self, model='yolov8n.pt'):
        super().__init__(model)

        # Model yapısını güçlendir
        # Sabit kanal sayıları kullanıyoruz (YOLOv8n için standart değerler)
        c2f_1_channels = 256
        c2f_2_channels = 512
        sppf_channels = 512
        head_channels = 256

        # Katmanları güncelle
        self.model.model[4] = C2f(c2f_1_channels, c2f_1_channels, n=2, shortcut=True)  # C2f_1
        self.model.model[6] = C2f(c2f_2_channels, c2f_2_channels, n=2, shortcut=True)  # C2f_2

        # SPPF modülünü geliştir
        self.model.model[9] = SPPF(sppf_channels, sppf_channels, k=5)  # SPPF

        # Head kısmını güçlendir
        self.model.model[15] = C2f(head_channels, head_channels, n=2, shortcut=True)  # Head C2f

        # Özel attention modülü ekle
        self.model.model.insert(8, AttentionModule(512))  # Attention modülü

    def train(self, **kwargs):
        # Eğitim parametrelerini güncelle
        custom_args = {
            'epochs': 100,
            'batch': 16,
            'workers': 8,
            'device': 0,
            'optimizer': 'SGD',
            'lr0': 0.01,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3.0,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 0.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.0,
            'copy_paste': 0.0,
        }

        custom_args.update(kwargs)
        return super().train(**custom_args)

class AttentionModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, channels//8, 1),
            nn.SiLU(),
            nn.Conv2d(channels//8, channels, 1),
            nn.Sigmoid()
        )

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//8, 1),
            nn.SiLU(),
            nn.Conv2d(channels//8, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        sa = self.spatial_attention(x)
        ca = self.channel_attention(x)
        return x * sa * ca

def create_custom_model(data_yaml):
    # Özel modeli oluştur
    model = CustomYOLOv8('yolov8n.pt')

    # Eğitim konfigürasyonunu ayarla
    model.train(
        data=data_yaml,
        name='yolov8n_puff_eye',
    )

    return model

if __name__ == "__main__":
    # Kullanım örneği
    data_yaml = '/content/drive/MyDrive/Puffy_Eye/Data/data.yaml'
    model = create_custom_model(data_yaml)