import torch
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# Charger le modèle et ses dépendances ici...
class ConvBlock(nn.Module):
    """Apply convolution, batch normalization, and ReLU twice."""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.cn1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.activ1 = nn.ReLU(inplace=True)
        self.cn2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activ2 = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.cn1(x)
        x = self.bn1(x)
        x = self.activ1(x)
        x = self.cn2(x)
        x = self.bn2(x)
        return self.activ2(x)

class DownScale(nn.Module):
    """Downscaling with max-pool then ConvBlock."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = ConvBlock(in_channels, out_channels)
    
    def forward(self, x):
        x = self.pool(x)
        x = self.block(x)
        return x

class UpScale(nn.Module):
    """Upscaling then ConvBlock, takes two inputs."""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = ConvBlock(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = ConvBlock(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
class Unet(nn.Module):
    def __init__(self, n_channels, n_classes, start=32, bilinear=False):
        super(Unet, self).__init__()
        self.inc = ConvBlock(n_channels, start)
        self.down1 = DownScale(start, 2*start)
        self.down2 = DownScale(2*start, 4*start)
        self.down3 = DownScale(4*start, 8*start)
        factor = 2 if bilinear else 1
        self.down4 = DownScale(8*start, 16*start // factor)
        self.up1 = UpScale(16*start, 8*start // factor, bilinear)
        self.up2 = UpScale(8*start, 4*start // factor, bilinear)
        self.up3 = UpScale(4*start, 2*start // factor, bilinear)
        self.up4 = UpScale(2*start, start, bilinear)
        self.outc = nn.Conv2d(start, n_classes, kernel_size=1)
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

number_of_classes = 8



# Palette de couleurs pour chaque classe (exemple avec 8 classes)
palette = [
    (0, 0, 0),       # Classe 0 : noir (souvent utilisé pour l'arrière-plan)
    (128, 0, 0),     # Classe 1 : rouge
    (0, 128, 0),     # Classe 2 : vert
    (0, 0, 128),     # Classe 3 : bleu
    (128, 128, 0),   # Classe 4 : jaune
    (128, 0, 128),   # Classe 5 : violet
    (0, 128, 128),   # Classe 6 : cyan
    (255, 128, 0)    # Classe 7 : orange
]

def colorize_mask(mask, palette):
    """
    Convertit un masque de classe en une image en couleurs en utilisant la palette spécifiée.
    
    Parameters:
    - mask (numpy array): Masque de classes (grayscale).
    - palette (list of tuples): Palette de couleurs pour chaque classe.
    
    Returns:
    - Image colorisée en fonction des classes.
    """
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    
    for class_idx, color in enumerate(palette):
        color_mask[mask == class_idx] = color
    
    return Image.fromarray(color_mask)

def segment_image_from_static(model, image_path, output_path):
    """
    Utilise le modèle U-Net pour segmenter une image et sauvegarde le résultat colorisé.
    
    Parameters:
    - model (nn.Module): Le modèle de segmentation U-Net chargé.
    - image_path (str): Le chemin de l'image d'entrée.
    - output_path (str): Le chemin où sauvegarder l'image segmentée colorisée.
    """
    # Charger l'image
    image = Image.open(image_path)
    
    # Transformation pour correspondre aux attentes du modèle
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Redimensionner selon les exigences du modèle
        transforms.ToTensor(),          # Convertir en tenseur
    ])
    input_tensor = transform(image).unsqueeze(0)  # Ajouter une dimension pour le batch
    
    # Segmenter l'image avec le modèle en mode évaluation
    model.eval()
    with torch.no_grad():
        pred_mask = model(input_tensor)
    pred_mask = torch.argmax(pred_mask, dim=1).squeeze(0).cpu().numpy()
    
    # Coloriser le masque
    color_segmented_image = colorize_mask(pred_mask, palette)
    
    # Sauvegarder l'image segmentée colorisée
    color_segmented_image.save(output_path)
    print(f"Image segmentée colorisée sauvegardée sous {output_path}")

# Exemple d'utilisation
image_path = "static/1.jpg"       # Remplacez par le chemin de l'image d'entrée
output_path = "static/output_segmented.png" # Chemin pour l'image segmentée colorisée

# Charger le modèle et appeler la fonction pour segmenter et sauvegarder l'image
model_load = Unet(3, number_of_classes)       # Charger le modèle U-Net (configuré avec les poids pré-enregistrés)
model_load.load_state_dict(torch.load("Unet_categorie_sans_augmentation.pth"))
segment_image_from_static(model_load, image_path, output_path)
