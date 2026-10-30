# 🧊 Gaussian App : de la vidéo au Gaussian Splatting, puis découpe d'objets avec SAGA

Une seule application Linux qui fait tout :

```
vidéo (360° comprise) / photos / projet COLMAP / PLY
        │
        ▼
images → caméras (COLMAP) → modèle 3D Gaussian Splatting → PLY
        │
        ▼
découpe d'objets avec SAGA → scène seule + un .ply par objet (Unity, Meta Quest 3)
```

Tout s'installe **automatiquement, sans sudo** : conda, FFmpeg, COLMAP, PyTorch CUDA, SAGA et Segment Anything.

---

## Sommaire

1. [Démarrage rapide](#1-démarrage-rapide)
2. [Prérequis](#2-prérequis)
3. [Ce que l'app installe toute seule](#3-ce-que-lapp-installe-toute-seule)
4. [Ajouter une scène](#4-ajouter-une-scène)
5. [Découper des objets avec SAGA](#5-découper-des-objets-avec-saga)
6. [Récupérer le résultat](#6-récupérer-le-résultat)
7. [Cas particuliers : PLY seul, LichtFeld, 360 Splat Pro, SuperSplat](#7-cas-particuliers)
8. [Vidéos 360° (Insta360)](#8-vidéos-360-insta360)
9. [YOLO (optionnel) : effacer la personne qui filme](#9-yolo-optionnel--effacer-la-personne-qui-filme)
10. [Réglages](#10-réglages)
11. [Ligne de commande](#11-ligne-de-commande)
12. [Organisation des dossiers](#12-organisation-des-dossiers)
13. [Dépannage](#13-dépannage)

---

## 1. Démarrage rapide

```bash
git clone https://github.com/DocteurWu/GaussianApp.git
cd GaussianApp
bash gaussian_app.sh
```

Pas de `git` ? Sur GitHub, cliquez **Code → Download ZIP**, décompressez, ouvrez un terminal dans le dossier, puis lancez `bash gaussian_app.sh`.

Le navigateur s'ouvre sur **http://localhost:7860**. Ensuite :

1. Cliquez **⬇ Tout installer**. Le premier lancement prend 20 à 40 minutes ; sur un PC déjà installé, c'est presque instantané.
2. **Glissez** une vidéo, des photos, un dossier COLMAP ou un `.ply` dans la page.
3. Cliquez **✂ Découper un objet**, puis suivez le mode d'emploi affiché dans la fenêtre SAGA.
4. Cliquez **📦 Dossier final** : vous obtenez la scène seule et un `.ply` par objet.

> [!TIP]
> Lancez toujours avec `bash gaussian_app.sh` : pas besoin de `chmod +x`.

---

## 2. Prérequis

| | |
|---|---|
| Système | Linux 64 bits (testé sur Ubuntu) |
| GPU | NVIDIA avec pilote installé : `nvidia-smi` doit répondre. RTX 2080 / 2080 Ti suffisent |
| RAM | 16 Go conseillés (32 Go pour de grosses scènes 360°) |
| Disque | ~20 Go pour les outils, plus quelques Go par projet |
| Réseau | Internet au premier lancement |
| Droits | **aucun droit admin nécessaire** |

Seul le pilote NVIDIA demande les droits admin pour être installé. Tout le reste va dans votre dossier personnel.

---

## 3. Ce que l'app installe toute seule

| Outil | Où |
|---|---|
| Conda (Miniforge) | `~/miniforge3`, si aucun conda n'est trouvé |
| FFmpeg, COLMAP, git | environnement conda `gs_tools` |
| PyTorch 2.1 + CUDA 11.8, compilateur CUDA, gcc compatible | environnement conda `saga_env` |
| SAGA (SegAnyGAussians v2) | cloné dans `saga/` et compilé |
| Segment Anything ViT-H (2,5 Go) | `saga/third_party/segment-anything/sam_ckpt/` |
| YOLO (ultralytics + yolov8x-seg) | seulement si vous l'activez |

Il n'est pas obligatoire de cliquer **Tout installer** : chaque outil s'installe aussi au moment où il sert. L'app applique automatiquement à SAGA les corrections décrites au [§13](#13-pourquoi-saga-ne-marchait-pas-avant-et-ce-qui-a-été-corrigé).

---

## 4. Ajouter une scène

Glissez-déposez dans la page. Pour les gros fichiers déjà présents sur la machine, utilisez **Fichier déjà sur la machine…** : l'app les utilise sans les copier. Le type d'entrée est détecté automatiquement.

| Vous donnez… | L'app fait… |
|---|---|
| 🎬 une **vidéo** (.mp4 .mov .mkv .avi .webm) | extraction d'images (2 par seconde, sans plans flous ni répétés) → COLMAP → 3DGS → PLY |
| 🌐 une **vidéo 360°** (.insv brut ou export équirectangulaire 2:1) | découpe en vues classiques (voir §8) → COLMAP → 3DGS → PLY |
| 🖼️ des **photos** (dossier ou .zip) | COLMAP → 3DGS → PLY |
| 📐 un **projet COLMAP** (`images/` + `sparse/0/`) | réutilise les caméras → 3DGS → PLY |
| 📐 + ✨ un **projet COLMAP avec un PLY** entraîné dessus | rattache le PLY aux caméras, prêt pour SAGA |
| ✨ un **PLY seul** (même `.compressed.ply` SuperSplat) | fabrique des vues virtuelles, prêt pour SAGA (voir §7) |

Après l'import, choisissez ce qui doit s'enchaîner : *rien*, *créer le PLY*, *PLY + préparer SAGA*, ou *tout + ouvrir la découpe*.

La console en bas de page montre l'avancement : bouton ⛶ pour le plein écran, A−/A+ pour la taille du texte. Le bouton 🌓 change le thème (sombre ou clair).

---

## 5. Découper des objets avec SAGA

Cliquez **✂ Découper un objet** sur le projet. Si la scène n'est pas prête, l'app calcule d'abord :
1. les masques **SAM** de chaque image (~2 s par image sur RTX 2080) ;
2. les échelles 3D des masques ;
3. les **features SAGA**, environ 10 à 20 minutes.

Puis la fenêtre SAGA s'ouvre, avec son mode d'emploi intégré.

### Pour chaque objet

1. Dans le panneau **Control**, cochez `clickmode`.
2. Faites **1 à 3 clics droits** sur l'objet. **Pas plus** : au-delà, SAGA finit par sélectionner toute la scène.
3. Réglez `ScoreThres` (0,7 à 0,9) jusqu'à ce que **seul l'objet** reste coloré.
   `Scale` : petit pour des pièces fines, grand pour l'objet entier.
4. Cliquez `segment3d`. Le **% sélectionné** s'affiche en haut : un objet = quelques %.
5. Tapez un nom (ex. `canape`) sous `save as`, puis cliquez `save as`.
6. Pour l'objet suivant : `clear`, puis recommencez à l'étape 2.
7. Quand c'est fini, **fermez la fenêtre**.

Navigation : clic gauche glissé = tourner · molette = zoom · clic milieu glissé = déplacer.

> [!IMPORTANT]
> Sans `segment3d`, ou si la sélection est vide ou couvre plus de 95 % de la scène, l'app **refuse** la sauvegarde. Le but est d'éviter un objet qui contient en réalité toute la pièce.

---

## 6. Récupérer le résultat

Cliquez **📦 Dossier final** sur le projet. L'app crée :

```
output_ply/<projet>/
├── scene_complete.ply      la scène entière
├── scene_seule.ply         la scène SANS les objets découpés
├── objets/
│   ├── canape.ply
│   └── chaise.ply ...
└── LISEZMOI.txt            nombre de gaussiennes par fichier
```

- Tous les fichiers sont **dans le même repère** : importés ensemble dans Unity (UnityGaussianSplatting), ils se superposent exactement.
- Pour chaque objet, l'app ne garde que le **bloc principal**. SAGA attrape souvent quelques gaussiennes ailleurs dans la pièce ; ces morceaux sont retirés. Pour les garder : `GS_OBJ_MAIN_ONLY=0`.
- Les fichiers se téléchargent aussi directement depuis l'interface.

Le PLY de la scène entière est aussi exporté dans `output_ply/FINAL_<projet>.ply`. Avec `sh-degree 1`, il est 3 fois plus léger, ce qui convient au Meta Quest 3.

---

## 7. Cas particuliers

### PLY seul (sans photos)

SAGA a besoin de photos et de caméras. Avec un PLY seul, l'app **rend elle-même ~300 vues virtuelles** de la scène : la moitié en orbite autour (4 hauteurs, 2 distances), l'autre moitié depuis l'intérieur. Les vues vides sont écartées. SAM et SAGA travaillent ensuite sur ces vues. L'avantage : les vues sont **forcément dans le même repère que le PLY**.

### PLY entraîné ailleurs (LichtFeld Studio, 360 Splat Pro, Postshot…)

SAGA attend un modèle produit par son propre code : un fichier `cfg_args` et le PLY rangé dans `point_cloud/iteration_N/scene_point_cloud.ply`. Ces logiciels ne produisent pas ce format. **L'app génère `cfg_args` elle-même** (avec le degré SH lu dans le PLY) et range le PLY au bon endroit.

Importez le dossier COLMAP **et** le PLY ensemble. Condition : le PLY doit être dans **le même repère** que les caméras COLMAP. Si la découpe « tombe à côté », c'est que le logiciel a recentré la scène. Deux solutions :
- laisser l'app **réentraîner** le modèle depuis le COLMAP (import du dossier COLMAP seul) ;
- importer le **PLY seul** (vues virtuelles).

### PLY SuperSplat compressé (`*.compressed.ply`)

Il est décompressé automatiquement à l'import. Ce format ne garde souvent que la couleur de base, sans reflets : c'est sans effet sur la découpe.

---

## 8. Vidéos 360° (Insta360)

Détection automatique. La méthode suit le workflow *Traitement_Gaussian_SAGA* :

- **`.insv` brut** (2 objectifs fisheye) : 3 vues par objectif (centre, −45°, +45°), 100° de champ, soit 6 vues par instant.
- **Export équirectangulaire** (vidéo 2:1 d'Insta360 Studio) : 6 vues horizontales tous les 60°, 100° de champ. Option : une vue vers le haut et une vers le bas (`GS_360_POLES=1`).
- **Caméra retournée, tours à 3 hauteurs** (haut, milieu, pieds) : aucun souci, COLMAP ne dépend pas de l'orientation. Le « verrouillage de direction » d'Insta360 Studio marche aussi.
- Le nombre d'images par seconde est réduit automatiquement pour ne pas dépasser 480 vues (`GS_360_MAX_IMAGES`). COLMAP connaît l'optique exacte des vues et apparie chaque instant avec les suivants.

---

## 9. YOLO (optionnel) : effacer la personne qui filme

**Désactivé par défaut.** SAGA n'en a pas besoin : la découpe utilise SAM et vos clics.

Activez la case **Masquer la personne qui filme** si vous apparaissez dans les images, typiquement en 360°. Ce que fait YOLO :
- modèle `yolov8x-seg`, seuil 0,25, silhouette élargie de 35 px (mains, vêtements, perche) ;
- **double passage** : image normale, puis retournée de 180°, car YOLO rate les humains vus à l'envers ;
- les zones masquées sont ignorées par COLMAP **et** par l'entraînement 3DGS (équivalent du « Mask Mode : Ignore » de LichtFeld Studio) : pas de fantôme appris ;
- aperçus (silhouette en rouge) : bouton **👁 Masques YOLO** du projet.

---

## 10. Réglages

Carte **⚙️ Réglages** de l'interface, avec trois préréglages : **⚡ Essai rapide**, **Normal** et **💎 Qualité**. Les valeurs sont enregistrées dans `.gaussian_app/settings.env`.

| Réglage | Défaut | Effet |
|---|---|---|
| `GS_FPS` | 2 | images extraites par seconde de vidéo (1 à 3 conseillé) |
| `GS_MAX_FRAMES` | 250 | nombre max d'images, réparties sur toute la vidéo |
| `GS_MAX_IMAGE_SIZE` | 1600 | résolution max (COLMAP + entraînement) |
| `GS_ITERS` | 30000 | itérations 3DGS (7000 = essai rapide) |
| `GS_SH_DEGREE` | 1 | reflets : 1 = léger (Quest 3), 3 = complets |
| `GS_SAM_DOWNSAMPLE` | 2 | réduction des images pour SAM. Si la mémoire GPU manque, l'app réessaie seule à 4 puis 8 |
| `GS_SAGA_ITERS` | 10000 | itérations des features SAGA |
| `GS_SAGA_RES` | 1024 | largeur max des images chargées par SAGA (get_scale, features). 800 si le PC manque de RAM |
| `GS_SYNTH_VIEWS` | 300 | vues virtuelles pour un PLY seul |
| `GS_360` | auto | `auto`, `1` (forcer) ou `0` (désactiver) |
| `GS_360_MAX_IMAGES` / `GS_360_SIZE` / `GS_360_FOV` | 480 / 1280 / 100 | vues 360° : nombre, taille, champ |
| `GS_360_POLES` | 0 | ajoute les vues haut et bas |
| `GS_YOLO` / `GS_YOLO_CLASSES` | 0 / person | masquage YOLO (optionnel) |
| `GS_OBJ_MAIN_ONLY` | 1 | ne garder que le bloc principal de chaque objet |

Un réglage ne s'applique qu'aux étapes **pas encore calculées**. Pour refaire un projet avec d'autres réglages, utilisez **↺ Refaire depuis…** : l'app efface ses propres calculs à partir de cette étape, mais jamais vos vidéos, photos ou PLY importés.

Une variable d'environnement est prioritaire : `GS_ITERS=7000 bash gaussian_app.sh run salon`.

---

## 11. Ligne de commande

```bash
bash gaussian_app.sh                          # interface web (par défaut)
bash gaussian_app.sh menu                     # menu dans le terminal
bash gaussian_app.sh install                  # tout installer
bash gaussian_app.sh import ~/salon.mp4       # ou dossier, .zip, .ply  (--name N)
bash gaussian_app.sh run salon                # -> output_ply/FINAL_salon.ply
bash gaussian_app.sh run salon --saga         # + préparation SAGA
bash gaussian_app.sh gui salon                # ouvre la découpe SAGA (prépare si besoin)
bash gaussian_app.sh package salon            # -> output_ply/salon/ (scène + objets)
bash gaussian_app.sh export salon [masque] [--invert]
bash gaussian_app.sh reset salon train        # refaire depuis : frames | colmap | train | saga
bash gaussian_app.sh build                    # traite toutes les vidéos de input_video/
bash gaussian_app.sh clean in.ply [out.ply]   # retire les floaters
bash gaussian_app.sh settings                 # réglages actifs
bash gaussian_app.sh status                   # état des outils et des projets
bash gaussian_app.sh install-launcher         # raccourci dans le menu des applications
bash gaussian_app.sh help
```

---

## 12. Organisation des dossiers

```
GaussianApp/
├── gaussian_app.sh        moteur + ligne de commande
├── gaussian_app_ui.py     interface web (Python standard, aucune dépendance)
├── input_video/           vidéos à traiter avec "build"
├── saga/                  SAGA (téléchargé automatiquement)
├── workspace/<projet>/    calculs d'un projet
│   ├── input/                 images brutes (vidéo ou photos)
│   ├── masks/ yolo_preview/   masques YOLO (si activé)
│   ├── distorted/             COLMAP avec distorsion
│   ├── images/ sparse/0/      images et caméras corrigées (format 3DGS)
│   ├── model/                 modèle 3DGS + features SAGA + segmentation_res/
│   └── sam_masks/ mask_scales/
├── output_ply/            résultats
├── .gaussian_app/         réglages et état de l'installation
└── gaussian_app.log       journal complet
```

---


## 13. Dépannage

| Symptôme | Solution |
|---|---|
| `Aucun GPU NVIDIA détecté` | Vérifier `nvidia-smi`. Le pilote doit être installé par un admin |
| Le navigateur ne s'ouvre pas | Ouvrir **http://localhost:7860** à la main, ou `GS_UI_PORT=7861 bash gaussian_app.sh` si le port est pris |
| Pas d'écran (connexion SSH) | `bash gaussian_app.sh menu`, ou `ssh -L 7860:localhost:7860 …` puis ouvrir le navigateur sur son propre PC. La découpe SAGA a besoin d'un écran |
| CUDA out of memory pendant SAM | l'app réessaie seule (×4, puis ×8). Sinon : Réglages > SAGA > Réduction des images pour SAM |
| « RAM du PC saturée » / code 137 | Réglages > SAGA > Largeur des images pour SAGA = 800, puis relancer. Pour suivre la RAM en direct : `free -h` ou `htop` dans un autre terminal |
| Voir l'erreur complète d'une étape SAGA | `.gaussian_app/last_saga_run.txt` (dernier programme SAGA lancé) ou `gaussian_app.log` |
| COLMAP ne trouve pas assez de caméras | filmer plus lentement, augmenter `GS_FPS`, éviter les surfaces unies et les reflets |
| Installation cassée | supprimer `.gaussian_app/saga_ok`, puis **⬇ Tout installer** |
| Autre | envoyer `gaussian_app.log` à Claude |
