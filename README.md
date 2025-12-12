# README GTSDB
## Détection et classification de panneaux de signalisation.

# 1. Créer un environnement virtuel 
python -m venv venv

# 2. Activer l'environnement 
venv\scripts\activate.ps1

# 3. Installer les librairies nécessaires
pip install -r requirements.txt

# 4. Recupération des données 
Telecharger les images via ce [lien.](https://erda.ku.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/TrainIJCNN2013.zip)

Creer un dossier /data à la racine du projet.

Creer un dossier /Image dans le dossier /data.

Placer les images (de 00000.ppm à 00599.ppm) dans dossier /Image.

Placer gt.txt dans le dossier /data.

Executer main.py de la partie 1 pour recuperer les fichiers json.


