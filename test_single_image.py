import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from train_segmentation import dice_coef, dice_loss, bce_dice_loss

def predict_single_image(image_path, model_path='best_model.h5'):
    """
    Prédit le masque pour une seule image en utilisant le meilleur modèle (epoch 96)
    """
    print(f"Chargement du meilleur modèle : {model_path}")
    # Charger le modèle
    model = tf.keras.models.load_model(model_path, 
                                     custom_objects={'dice_coef': dice_coef,
                                                   'dice_loss': dice_loss,
                                                   'bce_dice_loss': bce_dice_loss})
    
    # Charger et prétraiter l'image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Garder l'image originale pour l'affichage
    original_size = image.shape[:2]
    display_image = image.copy()
    
    # Redimensionner pour le modèle
    image = cv2.resize(image, (128, 128))
    image = image.astype('float32') / 255.0
    
    # Faire la prédiction
    prediction = model.predict(np.expand_dims(image, axis=0))[0]
    
    # Redimensionner la prédiction à la taille originale
    prediction_resized = cv2.resize(prediction.squeeze(), 
                                  (original_size[1], original_size[0]))
    
    # Créer une superposition colorée
    overlay = display_image.copy()
    mask_colored = np.zeros_like(display_image)
    mask_colored[prediction_resized > 0.5] = [255, 0, 0]  # Rouge pour les zones segmentées
    
    # Superposer avec transparence
    overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
    
    # Afficher les résultats
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(display_image)
    plt.title('Image Originale')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(prediction_resized, cmap='gray')
    plt.title('Masque Prédit')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(overlay)
    plt.title('Superposition')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('prediction_result.png')
    plt.show()
    
    print("Prédiction terminée ! Résultat sauvegardé dans 'prediction_result.png'")
    
    # Calcul de la surface avec l'échelle correcte
    surface_pixels, surface_meters, pixel_size = calculate_area(prediction_resized, original_size[1])
    
    print(f"\nSurface calculée:")
    print(f"Largeur du bâtiment détecté en pixels : {np.sum(prediction_resized[prediction_resized.shape[0]//2, :] > 0.5)}")
    print(f"Échelle : 1 pixel = {pixel_size:.3f} mètres")
    print(f"En pixels : {surface_pixels:,} pixels")
    print(f"En mètres carrés : {surface_meters:.2f} m²")
    print(f"Surface réelle du lot selon le plan : 520.00 m²")
    print(f"Différence : {abs(520 - surface_meters):.2f} m² ({abs(100 - (surface_meters/520)*100):.1f}%)")

def calculate_area(mask, image_width_pixels):
    """
    Calcule la surface de la zone segmentée
    Args:
        mask: Le masque binaire prédit (numpy array)
        image_width_pixels: Largeur de l'image en pixels
    Returns:
        surface_pixels: Surface en pixels
        surface_meters: Surface en mètres carrés
    """
    # Dimensions réelles du plan en mètres (à ajuster selon le plan)
    REAL_WIDTH_METERS = 10.0  # Largeur réelle approximative du bâtiment

    # Calcul de l'échelle (mètres par pixel)
    # On utilise la largeur du bâtiment comme référence
    building_width_pixels = np.sum(mask[mask.shape[0]//2, :] > 0.5)  # Largeur du bâtiment en pixels
    if building_width_pixels == 0:  # Si aucun pixel n'est détecté sur cette ligne
        building_width_pixels = np.sum(mask > 0.5) / mask.shape[0]  # Estimation moyenne
    
    pixel_size_meters = REAL_WIDTH_METERS / building_width_pixels
    
    # Compte le nombre de pixels blancs (valeur 1) dans le masque
    surface_pixels = np.sum(mask > 0.5)
    
    # Calcule la surface en mètres carrés
    surface_meters = surface_pixels * (pixel_size_meters ** 2)
    
    return surface_pixels, surface_meters, pixel_size_meters

if __name__ == "__main__":
    # Chemin vers l'image à tester
    image_path = "Plan_De_MasseIMG/test1.jpg"
    
    predict_single_image(image_path)
