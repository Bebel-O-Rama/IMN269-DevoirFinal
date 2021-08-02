# Devoir Final - IMN269
Nicolas Auclair-Labbé - aucn2303 \
Ala Antabli - anta2801

Ce document contient les informations relatives à l'exécution du programme ainsi que les données utilisées pour la calibration des caméras stéréo. 

## Architectures et versions
- OS: Windows 10
- Python: Version 3.8.8
- Numpy: 1.21.1
- Opencv-Python: 4.5.3.56
- Matplotlib: 3.4.2

## Exécution du programme
Pour exécuter le programme avec Windows, il faut avoir un terminal d'ouvert à la position du projet (donc dans le dossier IMN269-DevoirFinal) et y entrer une des deux commandes suivantes : 
1. Si on a **une image** qui contient les deux prises de vue, il faut utiliser : \
   `python stereo_cam_app.py <Path_combined_img>`
2. Si on a deux prises de vues dans **deux images séparées**, il faut utiliser : \
`python stereo_cam_app.py <Path_left_img> <Path_right_img>`
   
**IMPORTANT** : Dans les deux cas, les *path* doivent être des chemins relatifs vers les fichiers d'images.
- `python` est juste le mot clé pour les commandes de Python avec Windows 10. Il est laissé à la discrétion de l'utilisateur de changer cet argument s'il n'est pas valide/nécessaire pour son *OS*
- `stereo_cam_app.py` est le nom du fichier qu'on veut exécuter
- `<Path_combined_img>` est le chemin relatif vers l'image contenant les deux prises de vues la gauche et la droite. Notre implantation tient compte que les deux prises de vues sont séparées par une coupure verticale.
- `<Path_left_img>` est le chemin relatif vers l'image de la caméra gauche.
- `<Path_right_img>` est le chemin relatif vers l'image de la caméra droite.


## Informations de calibration
1. Paramètres intrinsèques:
- Taille du pixel: 4860m * 3660m
- Distortion radiale: zero
- Diamètre de la lentille: 1/3 pouce ou 8.47mm
- Distance focale: 12mm

2. Paramètres extrinsèques:
- Distance entre les deux caméras: 60mm
- Angle entre les deux caméras: nil
- Distance entre le centre des deux caméras et le visage: 480mm

Plus d'information à propos de la caméra peut être trouvée ici : https://www.aliexpress.com/item/32879864272.html