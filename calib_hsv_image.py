import cv2

# Fonction pour éviter les erreurs avec les trackbars
def doNothing(x):
    pass

# Fonction principale pour calibrer les valeurs HSV
def calibrateHSV(image_path):
    # Charger l'image depuis le chemin fourni
    image = cv2.imread(image_path)
    if image is None:
        raise IOError("Cannot load image. Check the file path.")
    
    find_hsv_thresh(image)

# Fonction pour détecter les seuils HSV
def find_hsv_thresh(image):
    # Créer une fenêtre redimensionnable pour les trackbars
    cv2.namedWindow('Track Bars', cv2.WINDOW_NORMAL)

    # Ajouter des trackbars pour les seuils HSV
    cv2.createTrackbar('min_H', 'Track Bars', 0, 179, doNothing)
    cv2.createTrackbar('min_S', 'Track Bars', 0, 255, doNothing)
    cv2.createTrackbar('min_V', 'Track Bars', 0, 255, doNothing)

    cv2.createTrackbar('max_H', 'Track Bars', 0, 179, doNothing)
    cv2.createTrackbar('max_S', 'Track Bars', 0, 255, doNothing)
    cv2.createTrackbar('max_V', 'Track Bars', 0, 255, doNothing)

    # Convertir l'image en HSV
    resized_image = cv2.resize(image, (800, 600))
    hsv_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)

    # Initialiser les valeurs max sur les trackbars
    cv2.setTrackbarPos('max_H', 'Track Bars', 179)
    cv2.setTrackbarPos('max_S', 'Track Bars', 255)
    cv2.setTrackbarPos('max_V', 'Track Bars', 255)

    while True:
        # Lire les valeurs des trackbars
        min_H = cv2.getTrackbarPos('min_H', 'Track Bars')
        min_S = cv2.getTrackbarPos('min_S', 'Track Bars')
        min_V = cv2.getTrackbarPos('min_V', 'Track Bars')

        max_H = cv2.getTrackbarPos('max_H', 'Track Bars')
        max_S = cv2.getTrackbarPos('max_S', 'Track Bars')
        max_V = cv2.getTrackbarPos('max_V', 'Track Bars')

        # Appliquer le seuillage avec les valeurs HSV
        mask = cv2.inRange(hsv_image, (min_H, min_S, min_V), (max_H, max_S, max_V))

        # Afficher l'image originale et la masque
        cv2.imshow('Original Image', resized_image)
        cv2.imshow('Mask Image', mask)

        # Sortir de la boucle si 'q' est pressé
        key = cv2.waitKey(25)
        if key == ord('q'):
            break

    # Afficher les seuils finaux
    print(f"min_HSV: ({min_H}, {min_S}, {min_V})")
    print(f"max_HSV: ({max_H}, {max_S}, {max_V})")

    # Fermer les fenêtres
    cv2.destroyAllWindows()

# Chemin de l'image locale
IMAGE_PATH = "test.png"

# Exécuter la calibration
calibrateHSV(IMAGE_PATH)
