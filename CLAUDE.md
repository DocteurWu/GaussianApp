# Gaussian App : contexte pour Claude

Application Linux tout-en-un : vidéo / photos / COLMAP / PLY → 3D Gaussian Splatting → découpe d'objets avec SAGA.
- `gaussian_app.sh` : moteur et ligne de commande (`bash gaussian_app.sh help`).
- `gaussian_app_ui.py` : interface web, Python standard, port 7860.
- `README.md` : la notice complète.

Les utilisateurs parlent français et veulent des explications simples.

## Contraintes du PC du labo
- **Aucun droit admin** : jamais de `sudo` ni d'`apt`. Tout passe par conda dans `~/miniforge3` :
  - `gs_tools` : COLMAP, FFmpeg, git ;
  - `saga_env` : Python 3.10, PyTorch 2.1.2 + CUDA 11.8.
- Carte NVIDIA (RTX 2080 / 2080 Ti) avec peu de mémoire. La RAM est souvent la limite.
- SAGA est cloné dans `saga/` (branche v2). L'app le **corrige automatiquement** à chaque lancement : fonctions `patch_saga_*` dans `gaussian_app.sh`. Ne pas modifier `saga/` à la main : corriger plutôt le patch dans `gaussian_app.sh`.
- YOLO doit rester **optionnel** (désactivé par défaut).

## Où regarder en cas de bug
- `gaussian_app.log` (à la racine, à côté de `workspace/`) : journal complet de chaque commande.
- `.gaussian_app/last_saga_run.txt` : sortie complète du dernier programme SAGA ou LichtFeld lancé.
- `.gaussian_app/settings.env` : réglages enregistrés depuis l'interface.
- `bash gaussian_app.sh status` : outils installés et état des projets.
- `bash gaussian_app.sh hw` : carte graphique, RAM libre, valeurs « auto », LichtFeld trouvé ou non.
- Projets : `workspace/<projet>/`, avec `images/`, `sparse/0/`, `model/`, `sam_masks/`, `mask_scales/`.

## Étapes et causes d'échec connues
1. Entraînement 3DGS : `train_scene.py` de SAGA, ou LichtFeld Studio si installé (`GS_TRAINER`).
2. SAGA 1/3 `extract_segment_everything_masks.py` : mémoire GPU → nouvel essai auto avec une réduction plus forte.
3. SAGA 2/3 `get_scale.py` : code 137 = RAM saturée → masques lus un par un (patch `patch_saga_lowmem`), images réduites (`GS_SAGA_RES`).
4. SAGA 3/3 `train_contrastive_feature.py` : idem.
5. `saga_gui.py` : la découpe a besoin d'un écran (pas en SSH sans affichage).

Pour relancer une étape : `bash gaussian_app.sh reset <projet> <frames|colmap|train|saga>`, puis `bash gaussian_app.sh run <projet> --saga`.
