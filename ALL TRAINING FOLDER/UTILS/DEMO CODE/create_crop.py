from PIL import Image
import numpy as np

def crop_center(image, crop_width=360, crop_height=240):
    # Se l'input è un array numpy, convertilo in un oggetto PIL
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
        
    # Ottieni le dimensioni dell'immagine originale
    width, height = image.size
   
    # Calcola le coordinate del rettangolo di ritaglio
    if crop_width < width and crop_height < height:
        left = (width - crop_width) // 2
        top = (height - crop_height) // 2
        right = (width + crop_width) // 2
        bottom = (height + crop_height) // 2

        # Esegui il ritaglio
        cropped_image = image.crop((left, top, right, bottom))
    else:
        cropped_image = image
    return cropped_image