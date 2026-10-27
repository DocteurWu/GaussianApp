#!/usr/bin/env bash
# ==============================================================================
# GAUSSIAN APP - Application Linux tout-en-un pour 3D Gaussian Splatting
#
#   Video / photos / projet COLMAP / PLY  ->  PLY 3DGS  ->  decoupe SAGA
#
# Tout s'installe tout seul, sans droits root (Miniforge + conda dans $HOME) :
# FFmpeg, COLMAP, git, PyTorch CUDA, SAGA, Segment Anything.
#
# Usage :
#   bash gaussian_app.sh              -> interface web (navigateur)
#   bash gaussian_app.sh menu         -> menu dans le terminal
#   bash gaussian_app.sh help         -> toutes les commandes
# ==============================================================================

# Pas de "set -u" : les scripts d'activation conda utilisent des variables non definies
set -o pipefail

APP_NAME="Gaussian App"
SELF_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "$SELF_PATH")"

# ------------------------------------------------------------------------------
# Chemins (detection auto de la racine de la cle USB)
# ------------------------------------------------------------------------------
if [ -d "$SCRIPT_DIR/input_video" ]; then
    BASE_DIR="$SCRIPT_DIR"
elif [ -d "$SCRIPT_DIR/../input_video" ] || [ -d "$SCRIPT_DIR/../output_ply" ]; then
    BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
else
    BASE_DIR="$PWD"
fi

SRC_VIDEO_DIR="$BASE_DIR/input_video"
WORK_ROOT="$BASE_DIR/workspace"
OUT_DIR="$BASE_DIR/output_ply"
SAGA_DIR="$BASE_DIR/saga"
SAM_MODEL="$SAGA_DIR/third_party/segment-anything/sam_ckpt/sam_vit_h_4b8939.pth"
STATE_DIR="$BASE_DIR/.gaussian_app"
SETTINGS_FILE="$STATE_DIR/settings.env"
MASTER_LOG="$BASE_DIR/gaussian_app.log"

# ------------------------------------------------------------------------------
# Parametres : variable d'environnement > reglages de l'interface (settings.env) > defaut
# ------------------------------------------------------------------------------
if [ -f "$SETTINGS_FILE" ]; then
    while IFS='=' read -r key value; do
        # Lecture stricte (jamais de "source") : GS_NOM=valeur simple uniquement
        [[ "$key" =~ ^GS_[A-Z_]+$ && "$value" =~ ^[A-Za-z0-9.,\ _-]*$ ]] || continue
        [ -n "${!key+x}" ] || printf -v "$key" '%s' "$value"
    done <"$SETTINGS_FILE"
fi
GS_FPS="${GS_FPS:-2}"                          # images extraites par seconde de video
GS_MAX_FRAMES="${GS_MAX_FRAMES:-250}"          # nombre max d'images gardees par scene
GS_ITERS="${GS_ITERS:-30000}"                  # iterations d'entrainement 3DGS
GS_SH_DEGREE="${GS_SH_DEGREE:-1}"              # 1 = optimise Meta Quest 3 (fichier 3x plus leger)
GS_MAX_IMAGE_SIZE="${GS_MAX_IMAGE_SIZE:-1600}" # resolution max des images (COLMAP + 3DGS)
GS_SAM_DOWNSAMPLE="${GS_SAM_DOWNSAMPLE:-2}"    # reduction des images pour SAM (memoire GPU)
GS_SAGA_ITERS="${GS_SAGA_ITERS:-10000}"        # iterations des features SAGA
GS_SYNTH_VIEWS="${GS_SYNTH_VIEWS:-300}"        # vues virtuelles rendues pour un PLY seul
GS_YOLO="${GS_YOLO:-0}"                        # 1 = YOLO masque la personne qui filme (optionnel)
GS_YOLO_CLASSES="${GS_YOLO_CLASSES:-person}"   # classes COCO a masquer (anglais, separees par des virgules)
GS_360="${GS_360:-auto}"                       # auto = .insv et videos 2:1 ; on / off pour forcer
GS_360_MAX_IMAGES="${GS_360_MAX_IMAGES:-480}"  # nombre max de vues decoupees au total
GS_360_SIZE="${GS_360_SIZE:-1280}"             # taille des vues decoupees (pixels, carre)
GS_360_FOV="${GS_360_FOV:-100}"                # champ de chaque vue (equirectangulaire)
GS_360_POLES="${GS_360_POLES:-0}"              # 1 = ajoute une vue vers le haut et une vers le bas
GS_UI_PORT="${GS_UI_PORT:-7860}"
SAGA_ENV_NAME="${SAGA_ENV_NAME:-saga_env}"
TOOLS_ENV_NAME="${TOOLS_ENV_NAME:-gs_tools}"

SAGA_REPO="https://github.com/Jumpat/SegAnyGAussians.git"
SAM_URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
YOLO_URL="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x-seg.pt"
MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
VIDEO_EXT_RE='\.(mp4|mov|mkv|avi|webm|m4v|insv)$'
IMAGE_EXT_RE='\.(jpg|jpeg|png)$'

mkdir -p "$SRC_VIDEO_DIR" "$WORK_ROOT" "$OUT_DIR" "$STATE_DIR" 2>/dev/null
export PATH="$HOME/.local/bin:$PATH"
export QT_QPA_PLATFORM=offscreen   # COLMAP sans ecran
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

# ------------------------------------------------------------------------------
# Utilitaires
# ------------------------------------------------------------------------------
hr()    { echo "======================================================================"; }
info()  { echo "[INFO] $*"; }
ok()    { echo "[OK] $*"; }
warn()  { echo "[WARN] $*"; }
err()   { echo "[ERROR] $*" >&2; }
die()   { err "$*"; exit 1; }
title() { hr; echo "$*"; hr; }
step()  { echo ""; echo "▶▶ $*"; }

start_logging() {
    exec > >(tee -a "$MASTER_LOG") 2>&1
    echo ""
    title "🚀 [$(date '+%Y-%m-%d %H:%M:%S')] $APP_NAME : $*"
}

safe_name() { printf '%s' "$1" | tr -cs 'a-zA-Z0-9._-' '_'; }
legacy_name() { echo "$1" | tr -cs 'a-zA-Z0-9._-' '_'; }   # noms de l'ancien build_scene.sh (finissent par _)

count_images() {
    find "$1" -maxdepth 1 -type f 2>/dev/null | grep -Eic "$IMAGE_EXT_RE"
}

count_files() {   # count_files <dossier> <extension>
    find "$1" -maxdepth 1 -type f -name "*.$2" 2>/dev/null | wc -l
}

download() {
    # Telechargement avec reprise : curl, puis wget, puis Python
    local url="$1" out="$2"
    mkdir -p "$(dirname "$out")"
    info "Telechargement : $url"
    if command -v curl &>/dev/null && curl -fL --retry 3 -C - -o "$out.part" "$url"; then
        mv -f "$out.part" "$out"; return 0
    fi
    if command -v wget &>/dev/null && wget -c -O "$out.part" "$url"; then
        mv -f "$out.part" "$out"; return 0
    fi
    python3 - "$url" "$out.part" <<'PY' && mv -f "$out.part" "$out"
import shutil, sys, urllib.request
with urllib.request.urlopen(sys.argv[1]) as r, open(sys.argv[2], 'wb') as f:
    shutil.copyfileobj(r, f, 1 << 20)
PY
}

# ------------------------------------------------------------------------------
# GPU
# ------------------------------------------------------------------------------
gpu_compute_cap() {
    nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d ' '
}

show_gpu() {
    if command -v nvidia-smi &>/dev/null && nvidia-smi -L &>/dev/null; then
        nvidia-smi --query-gpu=gpu_name,driver_version,memory.total --format=csv,noheader
        return 0
    fi
    warn "Aucun GPU NVIDIA detecte (nvidia-smi absent ou pilote inactif)."
    return 1
}

require_gpu() {
    show_gpu >/dev/null || die "Un GPU NVIDIA avec son pilote est obligatoire (l'installation du pilote demande les droits admin)."
}

detect_cuda_arch() {
    # Architecture reelle du GPU (ex: 8.6) pour compiler les extensions CUDA
    local cap
    cap="$(gpu_compute_cap)"
    export TORCH_CUDA_ARCH_LIST="${GS_CUDA_ARCH:-${cap:-7.5}}"
}

# ------------------------------------------------------------------------------
# Conda (Miniforge installe automatiquement dans $HOME, sans root)
# ------------------------------------------------------------------------------
CONDA_BIN=""

find_conda() {
    local c
    for c in "${CONDA_EXE:-}" "$HOME/miniforge3/bin/conda" "$HOME/miniconda3/bin/conda" \
             "$HOME/anaconda3/bin/conda" "/opt/conda/bin/conda" "$(command -v conda 2>/dev/null)"; do
        if [ -n "$c" ] && [ -x "$c" ]; then
            CONDA_BIN="$c"
            return 0
        fi
    done
    return 1
}

ensure_conda() {
    if ! find_conda; then
        warn "Conda absent : installation de Miniforge dans ~/miniforge3 (sans droits root)..."
        local inst="$STATE_DIR/downloads/Miniforge3-Linux-x86_64.sh"
        download "$MINIFORGE_URL" "$inst" || die "Telechargement de Miniforge impossible (connexion internet ?)."
        bash "$inst" -b -p "$HOME/miniforge3" || die "Installation de Miniforge echouee."
        CONDA_BIN="$HOME/miniforge3/bin/conda"
    fi
    eval "$("$CONDA_BIN" shell.bash hook)" || die "Impossible d'initialiser conda ($CONDA_BIN)."
}

env_prefix() {
    [ -n "$CONDA_BIN" ] || find_conda || return 0
    "$CONDA_BIN" env list 2>/dev/null | awk -v n="$1" '$1 == n {print $NF}' | head -n 1
}

# ------------------------------------------------------------------------------
# Outils : FFmpeg, COLMAP, git (systeme > binaire de la cle > conda-forge)
# ------------------------------------------------------------------------------
setup_static_ffmpeg() {
    command -v ffmpeg &>/dev/null && return 0
    local static_dir
    static_dir=$(find "$SCRIPT_DIR" -maxdepth 1 -type d -name "ffmpeg-*-static" 2>/dev/null | head -n 1)
    if [ -z "$static_dir" ]; then
        local arch
        arch=$(find "$SCRIPT_DIR" -maxdepth 2 -name "ffmpeg-*-static.tar.xz" 2>/dev/null | head -n 1)
        [ -n "$arch" ] && tar -xf "$arch" -C "$SCRIPT_DIR/" 2>/dev/null
        static_dir=$(find "$SCRIPT_DIR" -maxdepth 1 -type d -name "ffmpeg-*-static" 2>/dev/null | head -n 1)
    fi
    [ -n "$static_dir" ] && [ -f "$static_dir/ffmpeg" ] || return 1
    # Copie sur le disque local : les cles exFAT/FAT32 sont souvent montees en noexec
    mkdir -p "$HOME/.local/bin"
    cp -f "$static_dir/ffmpeg" "$static_dir/ffprobe" "$HOME/.local/bin/" 2>/dev/null
    chmod 755 "$HOME/.local/bin/ffmpeg" "$HOME/.local/bin/ffprobe" 2>/dev/null
    command -v ffmpeg &>/dev/null
}

use_tools_env() {
    local p
    p="$(env_prefix "$TOOLS_ENV_NAME")"
    [ -n "$p" ] && [ -d "$p/bin" ] || return 1
    case ":$PATH:" in
        *":$p/bin:"*) ;;
        *) export PATH="$PATH:$p/bin" ;;   # en dernier : les outils systeme restent prioritaires
    esac
}

ensure_tools() {
    # ensure_tools ffmpeg colmap git ...
    local t missing=()
    [[ " $* " == *" ffmpeg "* ]] && setup_static_ffmpeg
    use_tools_env
    for t in "$@"; do
        command -v "$t" &>/dev/null || missing+=("$t")
    done
    if [ ${#missing[@]} -gt 0 ]; then
        info "Installation automatique de : ${missing[*]} (conda-forge, sans root)..."
        ensure_conda
        if [ -n "$(env_prefix "$TOOLS_ENV_NAME")" ]; then
            "$CONDA_BIN" install -y -n "$TOOLS_ENV_NAME" --override-channels -c conda-forge "${missing[@]}"
        else
            "$CONDA_BIN" create -y -n "$TOOLS_ENV_NAME" --override-channels -c conda-forge "${missing[@]}"
        fi
        use_tools_env
    fi
    for t in "$@"; do
        command -v "$t" &>/dev/null || die "$t introuvable meme apres installation."
        ok "$t : $(command -v "$t")"
    done
}

colmap_flag() {
    # Les noms d'options COLMAP changent selon les versions : on prend celui qui existe
    local cmd="$1"; shift
    local help f
    help="$(colmap "$cmd" -h 2>&1)"
    for f in "$@"; do
        if grep -q -- "--$f" <<<"$help"; then echo "--$f"; return 0; fi
    done
    return 1
}

# ------------------------------------------------------------------------------
# SAGA : depot, environnement conda, modele SAM
# ------------------------------------------------------------------------------
saga_cuda_env() {
    if [ -x "$CONDA_PREFIX/bin/nvcc" ]; then
        export CUDA_HOME="$CONDA_PREFIX"
        export PATH="$CUDA_HOME/bin:$PATH"
        export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
    fi
    detect_cuda_arch
    export FORCE_CUDA=1
    export MAX_JOBS="${MAX_JOBS:-4}"
}

saga_activate() {
    find_conda || return 1
    eval "$("$CONDA_BIN" shell.bash hook)" 2>/dev/null || return 1
    [ -n "$(env_prefix "$SAGA_ENV_NAME")" ] || return 1
    conda activate "$SAGA_ENV_NAME" 2>/dev/null || return 1
    saga_cuda_env
}

saga_missing() {
    # Affiche les modules manquants de l'environnement SAGA (vide = tout est OK)
    python - <<'PY' 2>/dev/null || echo "python"
import importlib
mods = ["torch", "torchvision", "simple_knn", "diff_gaussian_rasterization",
        "diff_gaussian_rasterization_depth", "diff_gaussian_rasterization_contrastive_f",
        "segment_anything", "dearpygui", "plyfile", "cv2", "hdbscan", "sklearn", "scipy", "tqdm", "PIL"]
missing = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception:
        missing.append(m)
try:
    import torch
    if not torch.cuda.is_available():
        missing.append("cuda")
except Exception:
    pass
# versions incompatibles = a reparer (NumPy 2 avec PyTorch 2.1, dearpygui 2.x)
try:
    import numpy, torch
    if int(numpy.__version__.split(".")[0]) >= 2 and torch.__version__.startswith("2.1"):
        missing.append("numpy<2")
except Exception:
    pass
try:
    import dearpygui
    if int(dearpygui.__version__.split(".")[0]) >= 2:
        missing.append("dearpygui<2")
except Exception:
    pass
print(" ".join(missing))
PY
}

ensure_saga_repo() {
    # Les sous-modules SAGA sont declares en SSH (git@github.com:) : sans cle SSH ca echoue,
    # on force donc le HTTPS. segment-anything n'est pas clone ici : il est installe par pip.
    local https=(-c safe.directory='*' -c url."https://github.com/".insteadOf="git@github.com:")
    if [ ! -f "$SAGA_DIR/train_scene.py" ]; then
        ensure_tools git
        info "Telechargement de SAGA (SegAnyGAussians, branche v2)..."
        git "${https[@]}" clone -b v2 "$SAGA_REPO" "$SAGA_DIR" || die "git clone de SAGA echoue."
    fi
    local d missing=0
    for d in simple-knn diff-gaussian-rasterization diff-gaussian-rasterization-depth diff-gaussian-rasterization_contrastive_f; do
        [ -f "$SAGA_DIR/submodules/$d/setup.py" ] || missing=1
    done
    if [ "$missing" = 1 ]; then
        ensure_tools git
        git "${https[@]}" -C "$SAGA_DIR" submodule update --init --recursive -- submodules \
            || warn "Sous-modules SAGA incomplets (dossier submodules/)."
    fi
}

ensure_sam_ckpt() {
    local size=0
    [ -f "$SAM_MODEL" ] && size=$(wc -c <"$SAM_MODEL")
    if [ "$size" -lt 2000000000 ]; then
        info "Telechargement du modele Segment Anything ViT-H (~2.5 Go)..."
        download "$SAM_URL" "$SAM_MODEL" || die "Telechargement du modele SAM echoue."
    fi
    ok "Modele SAM : $SAM_MODEL"
}

patch_saga_gui() {
    # Aide en francais dans la fenetre + seuil par defaut utilisable (0.0 selectionnait tout)
    local gui="$SAGA_DIR/saga_gui.py"
    [ -f "$gui" ] || return 0
    python3 - "$gui" <<'PY'
import re, sys
p = sys.argv[1]
src = open(p, encoding="utf-8", newline="").read()
if "GAUSSIAN_APP_HELP" in src:
    sys.exit(0)
nl = "\r\n" if "\r\n" in src else "\n"
src = re.sub(r'(add_slider_float\(label="ScoreThres",\s*default_value=)0\.0', r'\g<1>0.75', src)
src = src.replace('width=300, height=550', 'width=300, height=self.window_height')
anchor = re.search(r'^( *)dpg\.add_text\("Mouse position: click anywhere to start\. ", tag="pos_item"\)\r?\n', src, re.M)
if anchor:
    ind = anchor.group(1)
    helptext = ("1. Cocher 'clickmode' (plus bas)\\n"
                "2. CLIC DROIT sur l'objet a garder\\n"
                "3. Ajuster ScoreThres (~0.7-0.9) :\\n"
                "   seul l'objet doit rester colore\\n"
                "   Scale : petit = pieces, grand = objet\\n"
                "4. Cliquer 'segment3d'\\n"
                "5. Ecrire un nom puis 'save as'\\n"
                "6. Fermer la fenetre : le PLY decoupe\\n"
                "   arrive dans output_ply/\\n\\n"
                "Vue : clic gauche = tourner,\\n"
                "molette = zoom, clic milieu = deplacer\\n"
                "'clear' = recommencer")
    block = (f'{ind}# GAUSSIAN_APP_HELP{nl}'
             f'{ind}with dpg.collapsing_header(label="MODE D\'EMPLOI", default_open=True):{nl}'
             f'{ind}    dpg.add_text("{helptext}", wrap=290, color=(255, 220, 120)){nl}')
    src = src[:anchor.end()] + block + src[anchor.end():]
open(p, "w", encoding="utf-8", newline="").write(src)
print("[OK] Interface SAGA annotee (mode d'emploi integre).")
PY
}

patch_saga_loader() {
    # SAGA recharge les masques SAM de toutes les vues sur le GPU (ils sont sauves depuis CUDA) :
    # sur une carte 8 Go (RTX 2080) c'est l'"out of memory" assure. On les garde en RAM ;
    # l'entrainement les envoie deja au GPU une vue a la fois (.cuda()).
    local f="$SAGA_DIR/scene/dataset_readers.py"
    [ -f "$f" ] || return 0
    python3 - "$f" <<'PY'
import sys
p = sys.argv[1]
src = open(p, encoding="utf-8", newline="").read()
lines = src.splitlines(keepends=True)
for i, l in enumerate(lines):
    if "torch.load(os.path.join(" in l and "_folder" in l and "map_location" not in l and '".pt"))' in l:
        lines[i] = l.replace('".pt"))', '".pt"), map_location=\'cpu\')')
new = "".join(lines)
if new != src:
    open(p, "w", encoding="utf-8", newline="").write(new)
    print("[OK] SAGA : masques SAM gardes en RAM (economie de memoire GPU).")
PY
}

patch_saga_knn() {
    # SAGA cherche les 16 voisins de chaque gaussienne avec pytorch3d (tres long a compiler).
    # L'ancien remplacement (torch.cdist sur toute la scene) demande N^2 distances : 3,6 To pour
    # 1 million de gaussiennes. On utilise un arbre KD (scipy) : quelques secondes, peu de memoire.
    local f="$SAGA_DIR/scene/gaussian_model_ff.py"
    [ -f "$f" ] || return 0
    python3 - "$f" <<'PY'
import re, sys
p = sys.argv[1]
src = open(p, encoding="utf-8", newline="").read()
if "GAUSSIAN_APP_KNN" in src:
    sys.exit(0)
nl = "\r\n" if "\r\n" in src else "\n"
shim = nl.join([
    "# GAUSSIAN_APP_KNN : k plus proches voisins par arbre KD (remplace pytorch3d.ops.knn_points)",
    "class _GaKnnOps:",
    "    @staticmethod",
    "    def knn_points(p1, p2, K=1):",
    "        from collections import namedtuple",
    "        from scipy.spatial import cKDTree",
    "        q = p1[0].detach().float().cpu().numpy()",
    "        ref = p2[0].detach().float().cpu().numpy()",
    "        d, i = cKDTree(ref).query(q, k=K, workers=-1)",
    "        if K == 1:",
    "            d, i = d[:, None], i[:, None]",
    "        KNN = namedtuple('KNN', ['dists', 'idx'])",
    "        return KNN(dists=torch.from_numpy(d ** 2).float()[None].to(p1.device),",
    "                   idx=torch.from_numpy(i).long()[None].to(p1.device))",
    "class _GaPytorch3D:",
    "    ops = _GaKnnOps",
    "pytorch3d = _GaPytorch3D()",
    "",
])
# ancien faux pytorch3d (cdist) ou import d'origine
src, n = re.subn(r"class DummyPytorch3D:.*?pytorch3d = DummyPytorch3D\(\)\r?\n", lambda m: shim, src, flags=re.S)
if n == 0:
    src, n = re.subn(r"^import pytorch3d(\.ops)?[^\r\n]*\r?\n", lambda m: shim, src, count=1, flags=re.M)
if n == 0:
    m = re.search(r"^import torch\r?\n", src, re.M)
    if not m:
        sys.exit("[WARN] gaussian_model_ff.py inattendu : recherche de voisins non remplacee.")
    src = src[:m.end()] + shim + src[m.end():]
src = re.sub(r"^import pytorch3d(\.ops)?[^\r\n]*\r?\n", "", src, flags=re.M)   # imports restants
open(p, "w", encoding="utf-8", newline="").write(src)
print("[OK] SAGA : recherche de voisins rapide (arbre KD) a la place de pytorch3d.")
PY
}

patch_saga_training() {
    # "Mask Mode: Ignore" (comme LichtFeld Studio) dans l'entrainement 3DGS de SAGA :
    # si GA_MASK_DIR est defini, les pixels masques (noirs) ne comptent pas dans l'erreur.
    local f="$SAGA_DIR/train_scene.py"
    [ -f "$f" ] || return 0
    python3 - "$f" <<'PY'
import re, sys
p = sys.argv[1]
src = open(p, encoding="utf-8", newline="").read()
if "GAUSSIAN_APP_MASK" in src:
    sys.exit(0)
nl = "\r\n" if "\r\n" in src else "\n"
helper = nl.join([
    "",
    "# GAUSSIAN_APP_MASK : masques YOLO ignores pendant l'entrainement (pixels noirs = sans erreur)",
    "_GA_MASK_DIR = os.environ.get('GA_MASK_DIR', '')",
    "_GA_MASKS = {}",
    "def _ga_mask(name, ref):",
    "    if not _GA_MASK_DIR:",
    "        return None",
    "    if name not in _GA_MASKS:",
    "        m = None",
    "        for ext in ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'):",
    "            mp = os.path.join(_GA_MASK_DIR, name + ext + '.png')",
    "            if os.path.isfile(mp):",
    "                from PIL import Image",
    "                m = torch.from_numpy(np.asarray(Image.open(mp).convert('L')) > 127)",
    "                break",
    "        _GA_MASKS[name] = m",
    "    m = _GA_MASKS[name]",
    "    if m is None or bool(m.all()):",
    "        return None",
    "    m = m[None, None].float().cuda()",
    "    return torch.nn.functional.interpolate(m, size=ref.shape[-2:], mode='nearest')[0]",
    "",
])
anchor = re.search(r"^from arguments import[^\r\n]*\r?\n", src, re.M)
gt = re.search(r"^( *)gt_image = viewpoint_cam\.original_image\.cuda\(\)\r?\n", src, re.M)
if not anchor or not gt:
    sys.exit("[WARN] train_scene.py inattendu : masques non appliques a l'entrainement.")
ind = gt.group(1)
use = nl.join([
    f"{ind}_ga_m = _ga_mask(viewpoint_cam.image_name, gt_image)",
    f"{ind}if _ga_m is not None:  # pixels masques : rendu remplace par la photo -> aucune erreur, aucun gradient",
    f"{ind}    image = image * _ga_m + gt_image * (1 - _ga_m)",
    "",
])
src = src[:gt.end()] + use + src[gt.end():]
src = src[:anchor.end()] + helper + nl + src[anchor.end():]
open(p, "w", encoding="utf-8", newline="").write(src)
print("[OK] SAGA : entrainement 3DGS avec masques ignores (Mask Mode: Ignore).")
PY
}

patch_saga_gui_guard() {
    # Garde-fous dans la fenetre SAGA :
    #  - apres segment3d, affiche le nombre de gaussiennes gardees (et previent si c'est trop large)
    #  - "save as" refuse une selection vide, toute la scene, ou sans segment3d
    #    (sinon SAGA enregistre 100 % de la scene sans rien dire)
    local gui="$SAGA_DIR/saga_gui.py"
    [ -f "$gui" ] || return 0
    python3 - "$gui" <<'PY'
import re, sys
p = sys.argv[1]
src = open(p, encoding="utf-8", newline="").read()
if "GAUSSIAN_APP_GUARD" in src:
    sys.exit(0)
nl = "\r\n" if "\r\n" in src else "\n"
done = []
m = re.search(r"^( *)self\.engine\['feature'\]\.segment\(self\.score_pts_binary\)\r?\n", src, re.M)
if m:
    i = m.group(1)
    add = nl.join([
        f"{i}# GAUSSIAN_APP_GUARD : taille de la selection",
        f"{i}_n = int(self.score_pts_binary.sum()); _p = 100.0 * _n / max(self.score_pts_binary.numel(), 1)",
        f"{i}print(f'segment3d : {{_n}} gaussiennes gardees ({{_p:.1f}} %)')",
        f"{i}dpg.set_value('pos_item', f'Selection : {{_n}} gaussiennes ({{_p:.1f}} %)' + ('  -> TROP LARGE : clear, moins de clics, ScoreThres plus haut' if _p > 60 else ''))",
        "",
    ])
    src = src[:m.end()] + add + src[m.end():]
    done.append("taille")
m = re.search(r"^( *)save_mask = self\.engine\['scene'\]\._mask == self\.engine\['scene'\]\.segment_times \+ 1\r?\n", src, re.M)
if m:
    i = m.group(1)
    add = nl.join([
        f"{i}# GAUSSIAN_APP_GUARD : jamais de masque vide ou egal a toute la scene",
        f"{i}_sel, _tot = int(save_mask.sum()), save_mask.numel()",
        f"{i}if self.engine['scene'].segment_times == 0:",
        f"{i}    raise ValueError('Rien de decoupe : cliquez segment3d AVANT save as.')",
        f"{i}if _sel == 0 or _sel > 0.95 * _tot:",
        f"{i}    raise ValueError(f'Selection = {{_sel * 100 // max(_tot, 1)}} % de la scene : refusee. clear, puis 1 a 3 clics droits sur l objet, ScoreThres plus haut, segment3d.')",
        f"{i}print(f'Objet : {{_sel}} gaussiennes sur {{_tot}} ({{_sel * 100.0 / _tot:.1f}} %)')",
        "",
    ])
    src = src[:m.end()] + add + src[m.end():]
    src = src.replace("dpg.add_text('You should segment the 3D object before save it (click segment3d first).')",
                      "dpg.add_text(str(e) if isinstance(e, ValueError) else 'Cliquez segment3d avant save as.', wrap=400)")
    done.append("sauvegarde")
src = src.replace("2. CLIC DROIT sur l'objet a garder",
                  "2. CLIC DROIT sur 1 a 3 points de l'objet\\n   (PAS PLUS : sinon toute la scene est prise)")
src = src.replace("5. Ecrire un nom puis 'save as'",
                  "5. Verifier le % affiche en haut (objet = petit %)\\n   puis ecrire un nom et 'save as'\\n   Objet suivant : 'clear' puis 1.")
if done:
    open(p, "w", encoding="utf-8", newline="").write(src)
    print("[OK] Fenetre SAGA : garde-fous ajoutes (" + ", ".join(done) + ").")
else:
    print("[WARN] saga_gui.py inattendu : garde-fous non ajoutes.")
PY
}

patch_saga_sources() {
    # Desactive le controle strict de version CUDA et ajoute <cstdint> (gcc recents)
    python - "$SAGA_DIR" <<'PY'
import os, sys
root = sys.argv[1]
subs = ["simple-knn", "diff-gaussian-rasterization", "diff-gaussian-rasterization-depth",
        "diff-gaussian-rasterization_contrastive_f"]
for s in subs:
    d = os.path.join(root, "submodules", s)
    setup = os.path.join(d, "setup.py")
    if os.path.isfile(setup):
        code = open(setup).read()
        if "_check_cuda_version = lambda" not in code:
            open(setup, "w").write("import torch.utils.cpp_extension\n"
                                   "torch.utils.cpp_extension._check_cuda_version = lambda *a, **k: None\n" + code)
    for h in ("rasterizer_impl.h", "forward.h", "backward.h"):
        hp = os.path.join(d, "cuda_rasterizer", h)
        if os.path.isfile(hp):
            hc = open(hp).read()
            if "#include <cstdint>" not in hc:
                open(hp, "w").write("#include <cstdint>\n" + hc)
    sk = os.path.join(d, "simple_knn.cu")
    if os.path.isfile(sk):
        c = open(sk).read()
        if "#include <cfloat>" not in c:
            open(sk, "w").write("#include <cfloat>\n" + c)
PY
}

ensure_saga_compilers() {
    # nvcc de la meme version CUDA que PyTorch + gcc compatible, tout dans l'env conda
    local tv
    tv="$(python -c 'import torch; print(torch.version.cuda or "")' 2>/dev/null)"
    [ -n "$tv" ] || die "PyTorch sans CUDA : impossible de compiler les extensions SAGA."
    if [ ! -x "$CONDA_PREFIX/bin/nvcc" ]; then
        info "Installation du compilateur CUDA $tv dans l'environnement (sans root)..."
        conda install -y --override-channels -c "nvidia/label/cuda-$tv.0" -c conda-forge cuda-toolkit \
            || die "Installation de cuda-toolkit $tv echouee."
    fi
    # Version max de gcc supportee par nvcc
    local maxgcc=11 sysgcc
    case "$tv" in
        12.0|12.1|12.2|12.3) maxgcc=12 ;;
        12.*) maxgcc=13 ;;
    esac
    sysgcc="$(gcc -dumpversion 2>/dev/null | cut -d. -f1)"
    if [ ! -x "$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++" ] && { [ -z "$sysgcc" ] || [ "$sysgcc" -gt "$maxgcc" ]; }; then
        info "gcc systeme ${sysgcc:-absent} incompatible avec CUDA $tv : installation de gcc $maxgcc (conda-forge)..."
        conda install -y --override-channels -c conda-forge "gcc_linux-64=$maxgcc" "gxx_linux-64=$maxgcc" \
            || die "Installation de gcc $maxgcc echouee."
    fi
    conda deactivate; conda activate "$SAGA_ENV_NAME"   # recharge CC/CXX du compilateur conda
    saga_cuda_env
    command -v nvcc &>/dev/null || die "nvcc introuvable."
    nvcc --version | grep release
}

install_saga_env() {
    require_gpu
    if [ -z "$(env_prefix "$SAGA_ENV_NAME")" ]; then
        info "Creation de l'environnement '$SAGA_ENV_NAME' (Python 3.10)..."
        "$CONDA_BIN" create -y -n "$SAGA_ENV_NAME" --override-channels -c conda-forge python=3.10 pip \
            || die "Creation de l'environnement conda echouee."
    fi
    conda activate "$SAGA_ENV_NAME" || die "Activation de $SAGA_ENV_NAME impossible."
    saga_cuda_env
    cd "$SAGA_DIR" || die "Dossier SAGA introuvable."
    # Versions compatibles avec PyTorch 2.1 / SAGA (sans effet si deja en place) :
    #  - setuptools >= 81 n'a plus pkg_resources, dont torch.utils.cpp_extension a besoin pour compiler
    #  - NumPy 2 casse PyTorch 2.1 ; dearpygui 2.x change l'API utilisee par saga_gui.py
    pip install "setuptools<70" "numpy<2" wheel || die "Installation de setuptools/numpy echouee."

    if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        local cap major
        cap="$(gpu_compute_cap)"; major="${cap%%.*}"
        if [ "${major:-0}" -ge 10 ]; then   # RTX 50xx (Blackwell) : CUDA 12.8 obligatoire
            pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
        else
            pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
        fi
        pip install "numpy<2"   # torchvision tire NumPy 2, incompatible avec PyTorch 2.1
        python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null \
            || die "PyTorch ne voit pas le GPU (pilote NVIDIA trop ancien ?). Voir nvidia-smi."
    fi
    ok "PyTorch : $(python -c 'import torch; print(torch.__version__, "| CUDA", torch.version.cuda, "|", torch.cuda.get_device_name(0))')"

    python -c "import plyfile, tqdm, hdbscan, cv2, sklearn, scipy, dearpygui, matplotlib, PIL" 2>/dev/null || {
        info "Installation des bibliotheques Python..."
        pip install "numpy<2" "scipy<1.12" plyfile tqdm hdbscan matplotlib opencv-python-headless \
            scikit-learn "dearpygui<2" pillow "setuptools<70" wheel ninja
    }
    python -c "import dearpygui, sys; sys.exit(int(dearpygui.__version__.split('.')[0]) >= 2)" 2>/dev/null \
        || pip install "dearpygui<2"

    local pair mod dir need_build=false
    local exts=("simple_knn:submodules/simple-knn"
                "diff_gaussian_rasterization:submodules/diff-gaussian-rasterization"
                "diff_gaussian_rasterization_depth:submodules/diff-gaussian-rasterization-depth"
                "diff_gaussian_rasterization_contrastive_f:submodules/diff-gaussian-rasterization_contrastive_f")
    for pair in "${exts[@]}"; do
        python -c "import ${pair%%:*}" 2>/dev/null || need_build=true
    done
    if [ "$need_build" = true ]; then
        ensure_saga_compilers
        pip install -q ninja
        patch_saga_sources
        for pair in "${exts[@]}"; do
            mod="${pair%%:*}"; dir="${pair#*:}"
            python -c "import $mod" 2>/dev/null && continue
            info "Compilation de $mod (quelques minutes)..."
            pip install --no-build-isolation "./$dir" || die "Compilation de $mod echouee (voir le log ci-dessus)."
        done
    fi

    python -c "import segment_anything" 2>/dev/null || {
        if [ -f third_party/segment-anything/setup.py ]; then
            pip install --no-build-isolation -e third_party/segment-anything
        else
            pip install "git+https://github.com/facebookresearch/segment-anything.git"
        fi
    }
}

SAGA_READY=0

ensure_saga() {
    [ "$SAGA_READY" = 1 ] && return 0
    require_gpu
    ensure_conda
    ensure_saga_repo
    ensure_sam_ckpt
    patch_saga_gui
    patch_saga_gui_guard
    patch_saga_loader
    patch_saga_training
    patch_saga_knn
    if ! saga_activate || [ -n "$(saga_missing)" ]; then
        info "Installation / reparation de l'environnement SAGA..."
        install_saga_env
        saga_activate || die "Environnement SAGA inutilisable."
    fi
    local miss
    miss="$(saga_missing)"
    [ -z "$miss" ] || die "Modules SAGA manquants : $miss"
    touch "$STATE_DIR/saga_ok"
    SAGA_READY=1
    ok "SAGA operationnel (env $SAGA_ENV_NAME, arch CUDA $TORCH_CUDA_ARCH_LIST)."
}

saga_py() {
    # Lance un script SAGA depuis son dossier
    ( cd "$SAGA_DIR" && python "$@" )
}

# ------------------------------------------------------------------------------
# Helpers Python (bibliotheque standard uniquement)
# ------------------------------------------------------------------------------
camera_is_pinhole() {
    # 0 si le modele COLMAP du dossier est sans distorsion (exige par 3DGS)
    python3 - "$1" <<'PY'
import os, struct, sys
d = sys.argv[1]
models = []
b, t = os.path.join(d, "cameras.bin"), os.path.join(d, "cameras.txt")
NPARAMS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5, 10: 12}
if os.path.isfile(b):
    with open(b, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            _, model, _, _ = struct.unpack("<iiQQ", f.read(24))
            f.read(8 * NPARAMS.get(model, 0))
            models.append(model)
elif os.path.isfile(t):
    names = {"SIMPLE_PINHOLE": 0, "PINHOLE": 1}
    for line in open(t):
        if line.strip() and not line.startswith("#"):
            models.append(names.get(line.split()[1], 99))
sys.exit(0 if models and all(m in (0, 1) for m in models) else 1)
PY
}

ply_is_3dgs() {
    python3 - "$1" <<'PY'
import sys
with open(sys.argv[1], "rb") as f:
    head = f.read(20000).split(b"end_header")[0].decode("ascii", "ignore")
props = {l.split()[-1] for l in head.splitlines() if l.startswith("property")}
need = {"x", "y", "z", "f_dc_0", "opacity", "scale_0", "rot_0"}
sys.exit(0 if need <= props else 1)
PY
}

ply_is_compressed() {
    head -c 20000 "$1" 2>/dev/null | grep -aq "packed_position"
}

decompress_ply() {
    # decompress_ply <in.compressed.ply> <out.ply> : format compresse SuperSplat/PlayCanvas -> PLY 3DGS standard
    local py=python3
    if ! python3 -c "import numpy" 2>/dev/null; then
        saga_activate || ensure_saga
        py=python
    fi
    "$py" - "$1" "$2" <<'PY'
import sys
import numpy as np
src, dst = sys.argv[1], sys.argv[2]
TYPES = {"float": "<f4", "float32": "<f4", "uint": "<u4", "uint32": "<u4", "uchar": "u1", "uint8": "u1"}
data = open(src, "rb").read()
end = data.index(b"end_header") + len(b"end_header")
end = data.index(b"\n", end) + 1
elems, cur = [], None
for line in data[:end].decode("ascii", "ignore").splitlines():
    p = line.split()
    if p and p[0] == "element":
        cur = [p[1], int(p[2]), []]; elems.append(cur)
    elif p and p[0] == "property":
        cur[2].append((p[-1], TYPES[p[1]]))
off, E = end, {}
for name, n, props in elems:
    dt = np.dtype(props)
    E[name] = np.frombuffer(data, dtype=dt, count=n, offset=off)
    off += dt.itemsize * n
ch, v = E["chunk"], E["vertex"]
n = len(v)
ci = np.arange(n) // 256

def lerp(lo, hi, t):
    return ch[lo][ci] + (ch[hi][ci] - ch[lo][ci]) * t

def u111011(x):
    return (x >> 21) / 2047.0, ((x >> 11) & 0x3FF) / 1023.0, (x & 0x7FF) / 2047.0

px, py_, pz = u111011(v["packed_position"])
sx, sy, sz = u111011(v["packed_scale"])
c = v["packed_color"]
r, g, b, a = (c >> 24) / 255.0, ((c >> 16) & 0xFF) / 255.0, ((c >> 8) & 0xFF) / 255.0, (c & 0xFF) / 255.0
if "min_r" in ch.dtype.names:
    r, g, b = lerp("min_r", "max_r", r), lerp("min_g", "max_g", g), lerp("min_b", "max_b", b)
# Rotation "smallest three" : 2 bits = composante la plus grande, 3 x 10 bits pour les autres
q = v["packed_rotation"]
norm = 1.0 / (np.sqrt(2) * 0.5)
abc = [(((q >> s) & 0x3FF) / 1023.0 - 0.5) * norm for s in (20, 10, 0)]
m = np.sqrt(np.clip(1.0 - sum(x * x for x in abc), 0, None))
largest = q >> 30
rot = np.zeros((n, 4))
for k in range(4):
    sel = largest == k
    others = [j for j in range(4) if j != k]
    rot[sel, k] = m[sel]
    for j, comp in zip(others, abc):
        rot[sel, j] = comp[sel]
SH_C0 = 0.28209479177387814
a = np.clip(a, 1e-6, 1 - 1e-6)
fields = {
    "x": lerp("min_x", "max_x", px), "y": lerp("min_y", "max_y", py_), "z": lerp("min_z", "max_z", pz),
    "f_dc_0": (r - 0.5) / SH_C0, "f_dc_1": (g - 0.5) / SH_C0, "f_dc_2": (b - 0.5) / SH_C0,
}
if "sh" in E:   # harmoniques spheriques optionnelles (1 octet chacune)
    sh = E["sh"]
    for name in sh.dtype.names:
        fields[name] = (sh[name] / 256.0 - 0.5) * 8.0
fields.update({
    "opacity": -np.log(1.0 / a - 1.0),
    "scale_0": lerp("min_scale_x", "max_scale_x", sx), "scale_1": lerp("min_scale_y", "max_scale_y", sy),
    "scale_2": lerp("min_scale_z", "max_scale_z", sz),
    "rot_0": rot[:, 0], "rot_1": rot[:, 1], "rot_2": rot[:, 2], "rot_3": rot[:, 3],
})
out = np.empty(n, dtype=[(k, "<f4") for k in fields])
for k, val in fields.items():
    out[k] = val
header = "ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % n
header += "".join(f"property float {k}\n" for k in fields) + "end_header\n"
with open(dst, "wb") as f:
    f.write(header.encode("ascii"))
    f.write(out.tobytes())
print(f"[OK] PLY SuperSplat compresse decompresse : {n} gaussiennes"
      f"{' (sans harmoniques : couleurs sans reflets)' if 'sh' not in E else ''}")
PY
}

write_cfg_args() {
    # write_cfg_args <model_dir> <source_dir> <ply> : cfg_args lu par les scripts SAGA
    python3 - "$1" "$2" "$3" <<'PY'
import os, sys
model, source, ply = sys.argv[1:4]
with open(ply, "rb") as f:
    head = f.read(20000).split(b"end_header")[0].decode("ascii", "ignore")
n_rest = sum(1 for l in head.splitlines() if l.startswith("property") and "f_rest_" in l)
deg = next((d for d in range(4) if 3 * (d + 1) ** 2 - 3 == n_rest), 3)
cfg = (f"Namespace(allow_principle_point_shift=False, data_device='cpu', eval=False, feature_dim=32, "
       f"feature_model_path='', images='images', init_from_3dgs_pcd=False, model_path={os.path.abspath(model)!r}, "
       f"need_features=False, need_masks=False, resolution=-1, sh_degree={deg}, "
       f"source_path={os.path.abspath(source)!r}, white_background=False)")
os.makedirs(model, exist_ok=True)
open(os.path.join(model, "cfg_args"), "w").write(cfg)
print(f"[OK] cfg_args ecrit (sh_degree={deg})")
PY
}

subsample_frames() {
    # subsample_frames <dossier> <N> [renumber] : garde au plus N images regulierement espacees
    # (renumber : renomme frame_00001.jpg... sans trous, pour les sequences FFmpeg)
    python3 - "$1" "$2" "${3:-}" <<'PY'
import os, re, sys
d, n, renumber = sys.argv[1], int(sys.argv[2]), sys.argv[3] == "renumber"
files = sorted(f for f in os.listdir(d) if re.search(r"\.(jpe?g|png)$", f, re.I))
if len(files) > n:
    keep = {files[round(i * (len(files) - 1) / max(n - 1, 1))] for i in range(n)}
    for f in files:
        if f not in keep:
            os.remove(os.path.join(d, f))
    print(f"[INFO] {len(files)} images -> {len(keep)} conservees (max {n})")
    files = sorted(keep)
if renumber:
    for i, f in enumerate(files, 1):
        os.rename(os.path.join(d, f), os.path.join(d, f"tmp_{i:05d}.jpg"))
    for i in range(1, len(files) + 1):
        os.rename(os.path.join(d, f"tmp_{i:05d}.jpg"), os.path.join(d, f"frame_{i:05d}.jpg"))
PY
}

# ------------------------------------------------------------------------------
# Projets : workspace/<nom>/
#   source/ ou .source_video   video d'origine
#   input/                     images brutes (video ou photos)
#   distorted/                 base COLMAP + modele avec distorsion
#   images/ + sparse/0/        images et cameras sans distorsion (format 3DGS)
#   model/                     modele 3DGS + features SAGA + segmentation_res/
#   sam_masks/ mask_scales/    donnees SAGA
# ------------------------------------------------------------------------------
proj_path() { echo "$WORK_ROOT/$1"; }

require_project() {
    [ -n "$1" ] || die "Nom de projet manquant."
    [ -d "$(proj_path "$1")" ] || die "Projet introuvable : $1 (voir 'status')."
}

new_project() {
    # new_project <nom souhaite> -> cree un dossier libre et affiche son nom
    local base name i=2
    base="$(safe_name "$1")"
    name="$base"
    while [ -e "$WORK_ROOT/$name" ]; do
        name="${base}_$i"; i=$((i + 1))
    done
    mkdir -p "$WORK_ROOT/$name"
    echo "$name"
}

scene_iter() {
    find "$1/model/point_cloud" -name "scene_point_cloud.ply" 2>/dev/null \
        | sed -E 's#.*/iteration_([0-9]+)/.*#\1#' | sort -n | tail -n 1
}

feat_iter() {
    find "$1/model/point_cloud" -name "contrastive_feature_point_cloud.ply" 2>/dev/null \
        | sed -E 's#.*/iteration_([0-9]+)/.*#\1#' | sort -n | tail -n 1
}

project_video() {
    local S="$1"
    if [ -f "$S/.source_video" ] && [ -f "$(cat "$S/.source_video")" ]; then
        cat "$S/.source_video"; return
    fi
    find "$S/source" -maxdepth 1 -type f 2>/dev/null | grep -Ei "$VIDEO_EXT_RE" | head -n 1
}

colmap_ready() {
    [ -d "$1/sparse/0" ] && [ "$(count_images "$1/images")" -ge 3 ] && camera_is_pinhole "$1/sparse/0"
}

migrate_legacy() {
    # Ancien format build_scene.sh : colmap/sparse/0 + images dans images/ ou nerfstudio_data/images/
    local S="$1" d
    [ -d "$S/colmap/sparse/0" ] && [ ! -d "$S/distorted/sparse/0" ] || return 0
    info "Projet de l'ancien pipeline detecte : reutilisation de son calcul COLMAP."
    mkdir -p "$S/distorted/sparse" "$S/input"
    cp -r "$S/colmap/sparse/0" "$S/distorted/sparse/0"
    if [ "$(count_images "$S/input")" -lt 3 ]; then
        for d in "$S/images" "$S/nerfstudio_data/images"; do
            if [ "$(count_images "$d")" -ge 3 ]; then
                cp "$d"/* "$S/input/"
                break
            fi
        done
    fi
    [ -d "$S/sparse" ] && mv "$S/sparse" "$S/_legacy_sparse"
    [ -d "$S/images" ] && [ ! -e "$S/_legacy_images" ] && mv "$S/images" "$S/_legacy_images"
}

step_frames() {
    local S="$1" n vid
    n=$(count_images "$S/input")
    if [ "$n" -ge 10 ]; then
        info "$n images deja presentes dans input/."
        return 0
    fi
    vid="$(project_video "$S")"
    [ -n "$vid" ] || { err "Ni video ni images dans ce projet."; return 1; }
    ensure_tools ffmpeg
    mkdir -p "$S/input"
    rm -f "$S/input"/*
    if video_is_360 "$S" "$vid"; then
        step_frames_360 "$S" "$vid" || return 1
    else
        step "Extraction des images de la video (FFmpeg, $GS_FPS img/s, plans fixes supprimes)"
        ffmpeg -hide_banner -loglevel warning -i "$vid" -vf "fps=$GS_FPS,mpdecimate" -qscale:v 2 "$S/input/frame_%05d.jpg" \
            || { err "FFmpeg a echoue sur $vid"; return 1; }
        subsample_frames "$S/input" "$GS_MAX_FRAMES"
    fi
    n=$(count_images "$S/input")
    [ "$n" -ge 10 ] || { err "Seulement $n images extraites : video trop courte ?"; return 1; }
    ok "$n images extraites."
}

# Video 360 : decoupee en vues classiques (format attendu par COLMAP / 3DGS / SAGA).
#  - equirectangulaire (export Insta360 Studio, 2:1) : 6 vues a l'horizontale tous les 60 deg
#    (+ haut/bas en option). Une camera retournee a 180 deg garde son horizon : ca reste valable.
#  - .insv brut (2 objectifs fisheye) : 3 vues par objectif (centre, -45, +45), comme le README.
INSV_LENS_FOV=190   # champ d'un objectif Insta360 X4

views_360() {
    # affiche "lacet tangage" par vue
    local mode="$1"
    if [ "$mode" = insv ]; then
        printf '%s\n' "0 0" "-45 0" "45 0" "0 0" "45 0" "-45 0"
    else
        printf '%s\n' "0 0" "60 0" "120 0" "180 0" "-120 0" "-60 0"   # v360 : lacet entre -180 et 180
        [ "$GS_360_POLES" = 1 ] && printf '%s\n' "0 90" "0 -90"
    fi
}

video_is_360() {
    # 0 si video 360 ; ecrit le mode (equirect | insv) dans <projet>/.is360
    local S="$1" vid="$2" wh w h
    if [[ "${vid,,}" == *.insv ]]; then
        echo insv >"$S/.is360"; return 0
    fi
    case "$GS_360" in
        on|1) echo equirect >"$S/.is360"; return 0 ;;
        off|0) rm -f "$S/.is360"; return 1 ;;
    esac
    [ -s "$S/.is360" ] && return 0
    wh="$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 "$vid" 2>/dev/null | head -n 1)"
    w="${wh%,*}"; h="${wh#*,}"
    # rapport 2:1 (a 1 % pres) = equirectangulaire
    if [[ "$w" =~ ^[0-9]+$ && "$h" =~ ^[0-9]+$ ]] && [ "$h" -gt 0 ] && [ $((w * 100 / h)) -ge 198 ] && [ $((w * 100 / h)) -le 202 ]; then
        info "Video 360 detectee (${w}x${h}, equirectangulaire)."
        echo equirect >"$S/.is360"; return 0
    fi
    return 1
}

step_frames_360() {
    local S="$1" vid="$2" mode views n dur fps fc maps=() i yaw pitch fov="$GS_360_FOV"
    mode="$(cat "$S/.is360")"
    mapfile -t views < <(views_360 "$mode")
    n=${#views[@]}
    # FPS reel : celui des reglages, reduit si la video est longue (max GS_360_MAX_IMAGES vues au total)
    dur="$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$vid" 2>/dev/null | head -n 1)"
    fps="$(python3 -c "d=float('${dur:-0}' or 0); print(round(min($GS_FPS, ($GS_360_MAX_IMAGES / $n) / d) if d > 0 else $GS_FPS, 4))")"
    if [ "$mode" = insv ]; then
        fov=100
        if [ "$(ffprobe -v error -select_streams v -show_entries stream=index -of csv=p=0 "$vid" | wc -l)" -ge 2 ]; then
            fc="[0:v:0]fps=$fps,split=3[a0][a1][a2];[0:v:1]fps=$fps,split=3[a3][a4][a5];"
        else   # les 2 objectifs cote a cote dans un seul flux
            fc="[0:v:0]fps=$fps,split=2[L][R];[L]crop=iw/2:ih:0:0,split=3[a0][a1][a2];[R]crop=iw/2:ih:iw/2:0,split=3[a3][a4][a5];"
        fi
        step "Video .insv (2 fisheye) : $n vues par instant, $fps img/s, ${GS_360_SIZE}px"
        for i in "${!views[@]}"; do
            read -r yaw pitch <<<"${views[$i]}"
            fc+="[a$i]v360=input=fisheye:ih_fov=$INSV_LENS_FOV:iv_fov=$INSV_LENS_FOV:output=rectilinear:h_fov=$fov:v_fov=$fov:yaw=$yaw:pitch=$pitch:w=$GS_360_SIZE:h=$GS_360_SIZE:interp=cubic[o$i];"
        done
    else
        step "Video 360 equirectangulaire : $n vues par panorama, $fps img/s, ${GS_360_SIZE}px, ${fov} deg"
        fc="[0:v:0]fps=$fps,split=$n"
        for i in "${!views[@]}"; do fc+="[a$i]"; done
        fc+=";"
        for i in "${!views[@]}"; do
            read -r yaw pitch <<<"${views[$i]}"
            fc+="[a$i]v360=input=equirect:output=rectilinear:h_fov=$fov:v_fov=$fov:yaw=$yaw:pitch=$pitch:w=$GS_360_SIZE:h=$GS_360_SIZE:interp=cubic[o$i];"
        done
    fi
    for i in "${!views[@]}"; do maps+=(-map "[o$i]" -q:v 2 "$S/input/f%05d_v$i.jpg"); done
    echo "$fov" >"$S/.view_fov"
    ffmpeg -hide_banner -loglevel warning -y -i "$vid" -filter_complex "${fc%;}" "${maps[@]}" \
        || { err "Decoupe 360 echouee (FFmpeg)."; return 1; }
}

ensure_yolo() {
    ensure_saga
    python -c "import ultralytics" 2>/dev/null || {
        info "Installation de YOLO (ultralytics)..."
        # --no-deps : ne pas toucher a PyTorch / numpy / OpenCV deja installes pour SAGA
        pip install --no-deps "ultralytics==8.3.40" ultralytics-thop \
            && pip install pyyaml requests psutil py-cpuinfo pandas seaborn \
            || die "Installation de YOLO echouee."
    }
    YOLO_MODEL="$STATE_DIR/models/$(basename "$YOLO_URL")"
    if [ ! -f "$YOLO_MODEL" ] && ! download "$YOLO_URL" "$YOLO_MODEL"; then
        # repli : ultralytics telecharge lui-meme le modele dans le dossier courant
        ( cd "$STATE_DIR/models" && python -c "from ultralytics import YOLO; YOLO('$(basename "$YOLO_URL")')" ) \
            || die "Telechargement du modele YOLO echoue."
    fi
}

step_yolo() {
    # OPTIONNEL (reglage GS_YOLO) : masque la personne qui filme.
    # COLMAP ignore ces zones, et l'entrainement 3DGS ne les apprend pas (comme "Mask Mode: Ignore").
    local S="$1"
    [ "$GS_YOLO" = 1 ] || return 0
    [ -f "$S/.yolo_done" ] && return 0
    ensure_yolo
    step "YOLO : masquage de '$GS_YOLO_CLASSES' (double passage endroit + camera retournee)"
    python - "$YOLO_MODEL" "$S" "$GS_YOLO_CLASSES" <<'PY' || return 1
import os, re, sys
import cv2
import numpy as np
from ultralytics import YOLO
model_path, scene, classes = sys.argv[1:4]
model = YOLO(model_path)
wanted = {c.strip().lower() for c in classes.split(",") if c.strip()}
ids = [i for i, n in model.names.items() if n.lower() in wanted]
unknown = wanted - {model.names[i].lower() for i in ids}
if unknown:
    print(f"[WARN] Classes inconnues ignorees : {', '.join(sorted(unknown))} (noms COCO en anglais : person, car, dog...)")
if not ids:
    sys.exit("[ERROR] Aucune classe YOLO valide.")
src, mdir, pdir = (os.path.join(scene, d) for d in ("input", "masks", "yolo_preview"))
os.makedirs(mdir, exist_ok=True)
os.makedirs(pdir, exist_ok=True)
files = sorted(f for f in os.listdir(src) if re.search(r"\.(jpe?g|png)$", f, re.I))
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))   # mains, vetements, perche, bords flous

def detect(img):
    h, w = img.shape[:2]
    out = np.zeros((h, w), np.uint8)
    r = model(img, classes=ids, conf=0.25, verbose=False)[0]
    if r.masks is not None:
        for m in r.masks.data:
            out = np.maximum(out, cv2.resize(m.cpu().numpy().astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST))
    return out

masked = 0
for i, f in enumerate(files):
    img = cv2.imread(os.path.join(src, f))
    # YOLO connait les humains debout : 2e passage image retournee pour une camera a l'envers
    p = np.maximum(detect(img), cv2.rotate(detect(cv2.rotate(img, cv2.ROTATE_180)), cv2.ROTATE_180))
    keep = np.full(p.shape, 255, np.uint8)
    if p.any():
        keep[cv2.dilate(p, kernel, iterations=2) > 0] = 0
        masked += 1
    cv2.imwrite(os.path.join(mdir, f + ".png"), keep)      # nom attendu par COLMAP : <image>.png
    prev = img.copy()
    prev[keep == 0] = (0, 0, 255)
    prev = cv2.addWeighted(img, 0.6, prev, 0.4, 0)
    s = 480 / max(prev.shape[:2])
    cv2.imwrite(os.path.join(pdir, f), cv2.resize(prev, None, fx=s, fy=s), [cv2.IMWRITE_JPEG_QUALITY, 80])
    if (i + 1) % 25 == 0 or i + 1 == len(files):
        print(f"  {i + 1}/{len(files)} images analysees")
print(f"[OK] YOLO : silhouette masquee dans {masked}/{len(files)} images (apercus rouges dans yolo_preview/).")
PY
    touch "$S/.yolo_done"
}

best_colmap_model() {
    # Le sous-modele COLMAP qui contient le plus d'images
    local d best="" size bsize=0
    for d in "$1"/*/; do
        [ -f "$d/images.bin" ] || [ -f "$d/images.txt" ] || continue
        size=$(cat "$d"/images.* 2>/dev/null | wc -c)
        if [ "$size" -gt "$bsize" ]; then best="${d%/}"; bsize=$size; fi
    done
    echo "$best"
}

step_colmap() {
    local S="$1" n threads best
    migrate_legacy "$S"
    if colmap_ready "$S"; then
        ok "Cameras COLMAP deja calculees ($(count_images "$S/images") images)."
        return 0
    fi
    if [ -d "$S/sparse/0" ]; then   # modele avec distorsion importe au mauvais endroit
        if [ ! -d "$S/distorted/sparse/0" ]; then
            mkdir -p "$S/distorted/sparse"; mv "$S/sparse/0" "$S/distorted/sparse/0"
        else
            mv "$S/sparse" "$S/_old_sparse_$(date +%s)"
        fi
    fi
    n=$(count_images "$S/input")
    [ "$n" -ge 3 ] || { err "Pas assez d'images dans input/ ($n)."; return 1; }
    ensure_tools colmap
    threads=$(nproc 2>/dev/null || echo 4); [ "$threads" -gt 8 ] && threads=8   # anti-OOM SIFT

    best="$(best_colmap_model "$S/distorted/sparse")"
    if [ -z "$best" ]; then
        mkdir -p "$S/distorted/sparse"
        local gpu_ext gpu_match thr_ext thr_match size_ext
        gpu_ext=$(colmap_flag feature_extractor FeatureExtraction.use_gpu SiftExtraction.use_gpu)
        thr_ext=$(colmap_flag feature_extractor FeatureExtraction.num_threads SiftExtraction.num_threads)
        size_ext=$(colmap_flag feature_extractor SiftExtraction.max_image_size FeatureExtraction.max_image_size)
        if [ ! -f "$S/distorted/.features_ok" ]; then
            step_yolo "$S" || return 1
            n=$(count_images "$S/input")
            local mask_opt=() cam_opt=(--ImageReader.camera_model OPENCV)
            [ "$(count_files "$S/masks" png)" -gt 0 ] && mask_opt=(--ImageReader.mask_path "$S/masks")
            if [ -f "$S/.is360" ]; then
                # Vues 360 decoupees : optique parfaite connue (focale = (taille/2) / tan(champ/2))
                local fx
                fx="$(python3 -c "import math; print((${GS_360_SIZE}/2) / math.tan(math.radians($(cat "$S/.view_fov" 2>/dev/null || echo 100))/2))")"
                cam_opt=(--ImageReader.camera_model PINHOLE --ImageReader.camera_params "$fx,$fx,$((GS_360_SIZE / 2)),$((GS_360_SIZE / 2))")
            fi
            step "COLMAP 1/3 : points caracteristiques SIFT ($n images, $threads threads)"
            rm -f "$S/distorted/database.db"
            colmap feature_extractor \
                --database_path "$S/distorted/database.db" \
                --image_path "$S/input" \
                "${mask_opt[@]}" \
                --ImageReader.single_camera 1 \
                "${cam_opt[@]}" \
                ${gpu_ext:+$gpu_ext 0} ${thr_ext:+$thr_ext $threads} \
                ${size_ext:+$size_ext $GS_MAX_IMAGE_SIZE} || { err "feature_extractor a echoue."; return 1; }
            touch "$S/distorted/.features_ok"
        fi
        local matcher=sequential_matcher
        [ -f "$S/.unordered" ] && [ "$n" -le 300 ] && matcher=exhaustive_matcher
        gpu_match=$(colmap_flag "$matcher" FeatureMatching.use_gpu SiftMatching.use_gpu)
        thr_match=$(colmap_flag "$matcher" FeatureMatching.num_threads SiftMatching.num_threads)
        step "COLMAP 2/3 : appariement des images ($matcher)"
        local match_opt=() map_opt=()
        if [ -f "$S/.is360" ]; then
            # chaque instant = 6 a 8 vues voisines : on apparie avec les ~3 instants suivants
            match_opt=(--SequentialMatching.overlap $(( $(views_360 "$(cat "$S/.is360")" | wc -l) * 3 )))
            local fl
            for fl in Mapper.ba_refine_focal_length Mapper.ba_refine_principal_point Mapper.ba_refine_extra_params; do
                colmap_flag mapper "$fl" >/dev/null && map_opt+=("--$fl" 0)   # optique connue : ne pas l'ajuster
            done
        fi
        colmap "$matcher" --database_path "$S/distorted/database.db" "${match_opt[@]}" \
            ${gpu_match:+$gpu_match 0} ${thr_match:+$thr_match $threads} || { err "$matcher a echoue."; return 1; }
        step "COLMAP 3/3 : reconstruction des cameras (mapper, le plus long)"
        colmap mapper --database_path "$S/distorted/database.db" --image_path "$S/input" "${map_opt[@]}" \
            --output_path "$S/distorted/sparse" || { err "mapper a echoue."; return 1; }
        best="$(best_colmap_model "$S/distorted/sparse")"
    fi
    [ -n "$best" ] || { err "COLMAP n'a retrouve aucune camera. Filmez plus lentement, avec plus de recouvrement et de texture."; return 1; }

    step "Correction de la distorsion de l'objectif (image_undistorter)"
    rm -rf "$S/images" "$S/stereo" "$S/sparse"
    colmap image_undistorter --image_path "$S/input" --input_path "$best" --output_path "$S" \
        --output_type COLMAP --max_image_size "$GS_MAX_IMAGE_SIZE" || { err "image_undistorter a echoue."; return 1; }
    mkdir -p "$S/sparse/0"
    find "$S/sparse" -maxdepth 1 -type f -exec mv {} "$S/sparse/0/" \;
    colmap_ready "$S" || { err "Sortie COLMAP invalide."; return 1; }
    ok "Cameras calculees pour $(count_images "$S/images") images."
}

step_train() {
    local S="$1" it
    it="$(scene_iter "$S")"
    if [ -n "$it" ]; then
        ok "Modele 3DGS deja entraine (iteration $it)."
        return 0
    fi
    ensure_saga
    local mask_dir=""
    if [ "$(count_files "$S/masks" png)" -gt 0 ]; then
        mask_dir="$S/masks"
        info "Masques YOLO trouves : les zones masquees sont ignorees pendant l'entrainement."
    fi
    step "Entrainement 3D Gaussian Splatting ($GS_ITERS iterations, sh-degree $GS_SH_DEGREE)"
    ( export GA_MASK_DIR="$mask_dir"
      saga_py train_scene.py -s "$S" -m "$S/model" --sh_degree "$GS_SH_DEGREE" \
        --iterations "$GS_ITERS" --save_iterations "$GS_ITERS" --data_device cpu --quiet ) \
        || { err "Entrainement 3DGS echoue."; return 1; }
    [ -n "$(scene_iter "$S")" ] || { err "Aucun modele produit."; return 1; }
}

step_export_final() {
    local S="$1" name="$2" it final
    [ -f "$S/.imported_ply" ] && return 0
    it="$(scene_iter "$S")"
    [ -n "$it" ] || return 1
    final="$OUT_DIR/FINAL_${name}.ply"
    cp -f "$S/model/point_cloud/iteration_$it/scene_point_cloud.ply" "$final"
    ok "PLY final (Unity / casque VR) : $final ($(du -h "$final" | cut -f1))"
}

step_synthetic_views() {
    # PLY seul : on fabrique des photos + cameras en faisant le rendu du PLY lui-meme
    local S="$1" it
    colmap_ready "$S" && return 0
    it="$(scene_iter "$S")"
    ensure_saga
    step "PLY sans photos : rendu de ~$GS_SYNTH_VIEWS vues virtuelles (autour + interieur) pour SAGA"
    python - "$SAGA_DIR" "$S/model/point_cloud/iteration_$it/scene_point_cloud.ply" "$S" "$GS_SYNTH_VIEWS" <<'PY' || return 1
import math, os, sys
import numpy as np
import torch
saga_dir, ply, scene, n_views = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
sys.path.insert(0, saga_dir)
from types import SimpleNamespace
from PIL import Image
from plyfile import PlyData, PlyElement
from scene.gaussian_model import GaussianModel
from scene.cameras import MiniCam
from gaussian_renderer import render
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

v = PlyData.read(ply)["vertex"].data
names = v.dtype.names
n_rest = sum(1 for n in names if n.startswith("f_rest_"))
deg = next(d for d in range(4) if 3 * (d + 1) ** 2 - 3 == n_rest)
xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
alpha = 1 / (1 + np.exp(-v["opacity"]))
pts = xyz[alpha > 0.5] if (alpha > 0.5).sum() > 1000 else xyz
center = np.median(pts, 0)
radius = float(np.percentile(np.linalg.norm(pts - center, axis=1), 80))
sub = pts[:: max(1, len(pts) // 200000)] - center
_, vecs = np.linalg.eigh(np.cov(sub.T))
up, ax_a = vecs[:, 0], vecs[:, 2]          # plus petite variance = vertical (sol / plafond)
ax_b = np.cross(up, ax_a)

gs = GaussianModel(deg)
gs.load_ply(ply)
pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
bg = torch.zeros(3, device="cuda")
W, H = 1024, 768                             # assez pour SAM, raisonnable en memoire
fovx = math.radians(70)
fovy = 2 * math.atan(math.tan(fovx / 2) * H / W)
fx, fy = W / (2 * math.tan(fovx / 2)), H / (2 * math.tan(fovy / 2))
znear, zfar = max(radius * 1e-3, 1e-3), max(radius * 50, 100.0)

def look_at(eye, target):
    z = target - eye; z /= np.linalg.norm(z)
    x = np.cross(-up, z)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(ax_a, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])                  # monde -> camera (convention COLMAP)
    return R, -R @ eye

def direction(az, el):
    return math.cos(el) * (math.cos(az) * ax_a + math.sin(az) * ax_b) + math.sin(el) * up

poses = []
n_out = n_views // 2                         # moitie en orbite autour de la scene
rings = [(el, rf) for el in (-15, 10, 30, 55) for rf in (1.3, 2.0)]
per_ring = max(4, math.ceil(n_out / len(rings)))
for i, (el, rf) in enumerate(rings):
    for k in range(per_ring):
        d = direction(2 * math.pi * (k + 0.5 * (i % 2)) / per_ring, math.radians(el))
        poses.append(look_at(center + d * radius * rf, center))
# moitie depuis l'interieur (pieces, scenes 360) : depuis le centre et 4 points decales
ext_a = float(np.percentile(np.abs(sub @ ax_a), 80))
ext_b = float(np.percentile(np.abs(sub @ ax_b), 80))
eyes = [center] + [center + s * 0.4 * e * ax for s in (-1, 1) for e, ax in ((ext_a, ax_a), (ext_b, ax_b))]
per_dir = max(4, math.ceil((n_views - len(poses)) / (len(eyes) * 3)))
for j, eye in enumerate(eyes):
    for el in (-25, 0, 25):
        for k in range(per_dir):
            d = direction(2 * math.pi * (k + j / len(eyes)) / per_dir, math.radians(el))
            poses.append(look_at(eye, eye + d))

def rot2quat(R):
    K = np.array([[R[0,0]-R[1,1]-R[2,2], 0, 0, 0],
                  [R[1,0]+R[0,1], R[1,1]-R[0,0]-R[2,2], 0, 0],
                  [R[2,0]+R[0,2], R[2,1]+R[1,2], R[2,2]-R[0,0]-R[1,1], 0],
                  [R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1], R[0,0]+R[1,1]+R[2,2]]]) / 3.0
    w, V = np.linalg.eigh(K)
    q = V[[3, 0, 1, 2], np.argmax(w)]
    return -q if q[0] < 0 else q

img_dir, sp_dir = os.path.join(scene, "images"), os.path.join(scene, "sparse", "0")
os.makedirs(img_dir, exist_ok=True); os.makedirs(sp_dir, exist_ok=True)
proj = getProjectionMatrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy).transpose(0, 1).cuda()
lines, kept = [], 0
with torch.no_grad():
    for i, (R, t) in enumerate(poses):
        wv = torch.tensor(getWorld2View2(R.T, t)).transpose(0, 1).cuda()
        full = wv.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)
        img = render(MiniCam(W, H, fovy, fovx, znear, zfar, wv, full), gs, pipe, bg)["render"].clamp(0, 1)
        arr = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        # vue presque vide, ou camera noyee dans une gaussienne (image uniforme) : inutile pour SAM
        if (arr.max(2) > 10).mean() < 0.25 or arr.std() < 6:
            continue
        kept += 1
        name = f"view_{kept:04d}.jpg"
        Image.fromarray(arr).save(os.path.join(img_dir, name), quality=95)
        q = rot2quat(R)
        lines.append(f"{kept} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} 1 {name}\n\n")
if kept < 10:
    sys.exit(f"[ERROR] Seulement {kept} vues exploitables : ce PLY n'est pas adapte.")
open(os.path.join(sp_dir, "cameras.txt"), "w").write(f"1 PINHOLE {W} {H} {fx} {fy} {W/2} {H/2}\n")
open(os.path.join(sp_dir, "images.txt"), "w").writelines(lines)
open(os.path.join(sp_dir, "points3D.txt"), "w").write("")
p = xyz[:: max(1, len(xyz) // 100000)]
rgb = np.clip(0.5 + 0.28209479 * np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1), 0, 1)[:: max(1, len(xyz) // 100000)]
el = np.empty(len(p), dtype=[(c, "f4") for c in ("x", "y", "z", "nx", "ny", "nz")] + [(c, "u1") for c in ("red", "green", "blue")])
el["x"], el["y"], el["z"] = p[:, 0], p[:, 1], p[:, 2]
el["nx"] = el["ny"] = el["nz"] = 0
el["red"], el["green"], el["blue"] = (rgb[:, 0] * 255), (rgb[:, 1] * 255), (rgb[:, 2] * 255)
PlyData([PlyElement.describe(el, "vertex")]).write(os.path.join(sp_dir, "points3D.ply"))
open(os.path.join(scene, ".synthetic_views"), "w").write(str(kept))
print(f"[OK] {kept} vues virtuelles rendues.")
PY
}

step_sam() {
    local S="$1" ni nm
    ni=$(count_images "$S/images"); nm=$(count_files "$S/sam_masks" pt)
    if [ "$nm" -ge "$ni" ]; then ok "Masques SAM deja calcules ($nm)."; return 0; fi
    ensure_saga
    step "SAGA 1/3 : masques 2D Segment Anything ($ni images, peut etre long)"
    saga_py extract_segment_everything_masks.py --image_root "$S" --sam_checkpoint_path "$SAM_MODEL" \
        --sam_arch vit_h --downsample "$GS_SAM_DOWNSAMPLE" --downsample_type image \
        || { err "Extraction SAM echouee (memoire GPU ? essayez GS_SAM_DOWNSAMPLE=4)."; return 1; }
}

step_scale() {
    local S="$1" ni ns
    ni=$(count_images "$S/images"); ns=$(count_files "$S/mask_scales" pt)
    if [ "$ns" -ge "$ni" ]; then ok "Echelles 3D deja calculees."; return 0; fi
    ensure_saga
    step "SAGA 2/3 : echelles 3D des masques (get_scale)"
    saga_py get_scale.py --image_root "$S" --model_path "$S/model" || { err "get_scale echoue."; return 1; }
}

step_features() {
    local S="$1"
    if [ -n "$(feat_iter "$S")" ]; then ok "Features SAGA deja entrainees."; return 0; fi
    ensure_saga
    step "SAGA 3/3 : apprentissage des features 3D ($GS_SAGA_ITERS iterations)"
    saga_py train_contrastive_feature.py -m "$S/model" --iterations "$GS_SAGA_ITERS" --num_sampled_rays 1000 \
        || { err "Entrainement des features SAGA echoue."; return 1; }
    [ -n "$(feat_iter "$S")" ] || { err "Features SAGA non produites."; return 1; }
}

# ------------------------------------------------------------------------------
# Commandes
# ------------------------------------------------------------------------------
cmd_install() {
    title "⚙️  Installation automatique de tout ce qui manque"
    show_gpu
    ensure_tools ffmpeg colmap git
    ensure_saga
    touch "$STATE_DIR/tools_ok"
    title "🎉 Tout est installe et pret."
}

import_images_into() {
    # import_images_into <source> <dest> <move:0|1> : copie a plat des images (noms uniques)
    python3 - "$1" "$2" "$3" <<'PY'
import os, re, shutil, sys
src, dst, move = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
os.makedirs(dst, exist_ok=True)
files = [src] if os.path.isfile(src) else sorted(
    os.path.join(r, f) for r, _, fs in os.walk(src) for f in fs if re.search(r"\.(jpe?g|png)$", f, re.I))
n = 0
for f in files:
    name = os.path.basename(f)
    if os.path.exists(os.path.join(dst, name)):
        name = os.path.basename(os.path.dirname(f)) + "_" + name
    (shutil.move if move else shutil.copy2)(f, os.path.join(dst, name))
    n += 1
print(f"[OK] {n} images importees.")
PY
}

attach_ply() {
    # attach_ply <projet_dir> <ply> <move:0|1>
    local S="$1" ply="$2" move="$3" dest
    dest="$S/model/point_cloud/iteration_30000"
    mkdir -p "$dest"
    if ply_is_compressed "$ply"; then
        decompress_ply "$ply" "$dest/scene_point_cloud.ply" || die "Decompression de $(basename "$ply") impossible."
        [ "$move" = 1 ] && rm -f "$ply"
    else
        ply_is_3dgs "$ply" || die "$(basename "$ply") n'est pas un PLY Gaussian Splatting (champs opacity/scale/rot absents)."
        if [ "$move" = 1 ]; then mv -f "$ply" "$dest/scene_point_cloud.ply"; else cp -f "$ply" "$dest/scene_point_cloud.ply"; fi
    fi
    write_cfg_args "$S/model" "$S" "$dest/scene_point_cloud.ply"
    touch "$S/.imported_ply"
}

cmd_import() {
    local src="" name="" move=0
    while [ $# -gt 0 ]; do
        case "$1" in
            --name) name="$2"; shift ;;
            --move) move=1 ;;
            *) src="$1" ;;
        esac
        shift
    done
    [ -n "$src" ] && [ -e "$src" ] || die "Usage: import <video|dossier|archive.zip|fichier.ply> [--name NOM] [--move]"
    src="$(cd "$(dirname "$src")" && pwd)/$(basename "$src")"

    # Archive -> dossier temporaire
    if [ -f "$src" ] && [[ "${src,,}" =~ \.(zip|tar|tgz|tar\.gz|tar\.xz)$ ]]; then
        local ex="$STATE_DIR/extract/$(date +%s)_$$"
        info "Decompression de $(basename "$src")..."
        python3 - "$src" "$ex" <<'PY' || die "Archive illisible."
import shutil, sys
shutil.unpack_archive(sys.argv[1], sys.argv[2])
PY
        [ -n "$name" ] || name="$(basename "${src%.*}")"
        src="$ex"; move=1
    fi
    # Dossier qui ne contient qu'un seul fichier -> ce fichier
    if [ -d "$src" ] && [ "$(find "$src" -type f | wc -l)" -eq 1 ]; then
        src="$(find "$src" -type f)"
    fi

    local S P
    if [ -f "$src" ]; then
        [ -n "$name" ] || name="$(basename "${src%.*}")"
        if grep -Eiq "$VIDEO_EXT_RE" <<<"$src"; then
            P="$(new_project "$name")"; S="$(proj_path "$P")"
            if [ "$move" = 1 ]; then
                mkdir -p "$S/source"; mv "$src" "$S/source/"
            else
                echo "$src" >"$S/.source_video"   # pas de copie des grosses videos
            fi
            ok "Projet '$P' cree depuis la video $(basename "$src")."
        elif grep -Eiq '\.ply$' <<<"$src"; then
            P="$(new_project "$name")"; S="$(proj_path "$P")"
            attach_ply "$S" "$src" "$move"
            ok "Projet '$P' cree depuis le PLY (SAGA fabriquera des vues virtuelles)."
        else
            die "Type de fichier non reconnu : $(basename "$src")"
        fi
    elif [ -d "$src" ]; then
        [ -n "$name" ] || name="$(basename "$src")"
        local cams
        cams="$(find "$src" -type f \( -name cameras.bin -o -name cameras.txt \) | head -n 1)"
        if [ -n "$cams" ]; then
            # Projet COLMAP : images + sparse (+ eventuellement un PLY entraine dessus)
            local model imgdir parent best_n=0 d c
            model="$(dirname "$cams")"
            parent="$(dirname "$model")"; [ "$(basename "$parent")" = sparse ] && parent="$(dirname "$parent")"
            imgdir=""
            [ "$(count_images "$parent/images")" -ge 3 ] && imgdir="$parent/images"
            if [ -z "$imgdir" ]; then
                while IFS= read -r d; do
                    c=$(count_images "$d")
                    if [ "$c" -gt "$best_n" ]; then imgdir="$d"; best_n=$c; fi
                done < <(find "$src" -type d)
            fi
            [ -n "$imgdir" ] || die "Projet COLMAP sans images : ajoutez le dossier images/ avec le dossier sparse/."
            P="$(new_project "$name")"; S="$(proj_path "$P")"
            if camera_is_pinhole "$model"; then
                import_images_into "$imgdir" "$S/images" "$move"
                mkdir -p "$S/sparse"; cp -r "$model" "$S/sparse/0"
            else
                import_images_into "$imgdir" "$S/input" "$move"
                mkdir -p "$S/distorted/sparse"; cp -r "$model" "$S/distorted/sparse/0"
            fi
            local ply
            ply="$(find "$src" -type f -iname '*.ply' ! -name 'points3D.ply' ! -name 'sparse_pc.ply' ! -name 'input.ply' | head -n 1)"
            if [ -n "$ply" ]; then
                attach_ply "$S" "$ply" "$move"
                info "PLY $(basename "$ply") associe aux cameras COLMAP."
            fi
            touch "$S/.imported_colmap"
            ok "Projet '$P' cree depuis un projet COLMAP ($(basename "$imgdir"))."
        elif [ "$(find "$src" -type f | grep -Eic "$IMAGE_EXT_RE")" -ge 3 ]; then
            P="$(new_project "$name")"; S="$(proj_path "$P")"
            import_images_into "$src" "$S/input" "$move"
            touch "$S/.unordered"
            ok "Projet '$P' cree depuis des photos."
        else
            local f found=0
            while IFS= read -r f; do
                cmd_import "$f" $([ "$move" = 1 ] && echo --move)
                found=1
            done < <(find "$src" -type f | grep -Ei "$VIDEO_EXT_RE|\.ply$")
            [ "$found" = 1 ] || die "Rien d'exploitable dans $src (videos, photos, COLMAP ou PLY attendus)."
            return 0
        fi
    fi
    [ "$move" = 1 ] && [[ "$src" == "$STATE_DIR"/* ]] && rm -rf "$STATE_DIR/extract/"* 2>/dev/null
    echo "PROJECT=$P"
}

cmd_run() {
    local P="" saga=false gui=false
    while [ $# -gt 0 ]; do
        case "$1" in
            --saga) saga=true ;;
            --gui) saga=true; gui=true ;;
            *) P="$1" ;;
        esac
        shift
    done
    require_project "$P"
    local S; S="$(proj_path "$P")"
    title "🎬 Projet $P"

    if [ -f "$S/.imported_ply" ] && ! colmap_ready "$S" && [ "$(count_images "$S/input")" -lt 3 ] && [ -z "$(project_video "$S")" ]; then
        [ "$saga" = true ] || { ok "Projet PLY : rien a reconstruire. Utilisez la decoupe SAGA."; return 0; }
        step_synthetic_views "$S" || return 1
    else
        colmap_ready "$S" || [ -d "$S/distorted/sparse/0" ] || [ -d "$S/colmap/sparse/0" ] || step_frames "$S" || return 1
        step_colmap "$S" || return 1
        step_train "$S" || return 1
        step_export_final "$S" "$P"
    fi

    if [ "$saga" = true ]; then
        step_sam "$S" || return 1
        step_scale "$S" || return 1
        step_features "$S" || return 1
        ok "Scene prete pour la decoupe SAGA."
    fi
    [ "$gui" = true ] && cmd_gui "$P"
    return 0
}

cmd_reset() {
    # reset <projet> <frames|colmap|train|saga> : efface ce que l'app a calcule a partir de cette etape
    # (jamais la video, les photos ou le PLY importes) pour la refaire avec les nouveaux reglages
    local P="$1" from="$2" S
    require_project "$P"
    S="$(proj_path "$P")"
    case "$from" in
        frames)
            [ -n "$(project_video "$S")" ] || die "Seuls les projets video ont des images extraites a refaire."
            rm -rf "$S/input" "$S/equirect" "$S/.is360" "$S/.view_fov" ;;&
        frames|colmap)
            [ -f "$S/.imported_colmap" ] && die "Les cameras de ce projet ont ete importees : rien a recalculer."
            if [ -f "$S/.imported_ply" ] && [ ! -f "$S/.synthetic_views" ]; then
                die "Les cameras de ce projet viennent de l'import : rien a recalculer."
            fi
            if [ -d "$S/_legacy_images" ] || [ -d "$S/colmap/sparse/0" ]; then
                die "Ancien projet : son calcul COLMAP d'origine est conserve (importez la video a nouveau pour tout refaire)."
            fi
            rm -rf "$S/distorted" "$S/images" "$S/sparse" "$S/stereo" "$S/masks" "$S/yolo_preview" "$S/.yolo_done" "$S/.synthetic_views" \
                   "$S"/run-colmap-*.sh ;;&
        frames|colmap|train)
            if [ -f "$S/.imported_ply" ]; then
                [ "$from" = train ] && die "Le modele de ce projet est le PLY importe : rien a re-entrainer."
            else
                rm -rf "$S/model"
            fi
            rm -rf "$S/sam_masks" "$S/mask_scales" ;;&
        frames|colmap|train|saga)
            rm -f "$S"/model/point_cloud/iteration_*/contrastive_feature_point_cloud.ply \
                  "$S"/model/point_cloud/iteration_*/scale_gate.pt
            [ "$from" = saga ] && rm -rf "$S/sam_masks" "$S/mask_scales"
            # dossiers d'iteration devenus vides (sinon SAGA plante en les listant)
            find "$S/model/point_cloud" -mindepth 1 -maxdepth 1 -type d -empty -delete 2>/dev/null
            ;;
        *) die "Usage: reset <projet> <frames|colmap|train|saga>" ;;
    esac
    ok "Projet $P remis a l'etape '$from'. Relancez-le pour recalculer avec les reglages actuels."
}

export_mask() {
    # export_mask <scene_ply> <mask.pt> <out.ply> [invert:0|1]
    python - "$1" "$2" "$3" "${4:-0}" <<'PY'
import os, sys
import torch
from plyfile import PlyData, PlyElement
ply, mask_path, out, invert = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] == "1"
data = PlyData.read(ply)["vertex"].data
mask = torch.load(mask_path, map_location="cpu")
if isinstance(mask, dict) and "mask" in mask:
    mask = mask["mask"]
mask = mask.bool().numpy().reshape(-1)
if invert:
    mask = ~mask
if len(mask) != len(data):
    sys.exit(f"[ERROR] Masque ({len(mask)}) et PLY ({len(data)}) incompatibles.")
if not invert and (mask.sum() == 0 or mask.mean() > 0.95):
    sys.exit(f"[ERROR] {os.path.basename(mask_path)} selectionne {mask.mean()*100:.0f} % de la scene : "
             "ce n'est pas un objet (trop de clics ou save as sans segment3d). Refaites-le dans SAGA.")
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
PlyData([PlyElement.describe(data[mask], "vertex")]).write(out)
print(f"[OK] {out} : {int(mask.sum())}/{len(data)} gaussiennes")
PY
}

cmd_package() {
    # package <projet> : output_ply/<projet>/ = scene_complete.ply + scene_seule.ply (sans les objets)
    #                    + objets/<nom>.ply, uniquement avec des masques valides
    local P="$1" S it py=python3
    require_project "$P"
    S="$(proj_path "$P")"; it="$(scene_iter "$S")"
    [ -n "$it" ] || die "Pas encore de modele 3D pour $P."
    saga_activate 2>/dev/null && py=python
    "$py" -c "import numpy" 2>/dev/null || die "numpy introuvable (lancez d'abord l'installation)."
    title "📦 Dossier final : $P"
    "$py" - "$S/model/point_cloud/iteration_$it/scene_point_cloud.ply" "$S/model/segmentation_res" "$OUT_DIR/$P" <<'PY'
import glob, os, re, shutil, sys, unicodedata, zipfile
import numpy as np
scene, seg_dir, out = sys.argv[1:4]
T = {"float": "f4", "float32": "f4", "double": "f8", "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
     "short": "i2", "ushort": "u2", "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4"}

def read_ply(path):
    raw = open(path, "rb").read()
    end = raw.index(b"end_header") + len(b"end_header")
    end = raw.index(b"\n", end) + 1
    head = raw[:end].decode("ascii", "ignore").splitlines()
    if not any(l.startswith("format binary_little_endian") for l in head):
        sys.exit("[ERROR] PLY non binaire : non supporte.")
    props, n, cur = [], 0, None
    for l in head:
        p = l.split()
        if p[:1] == ["element"]:
            cur = p[1]
            if cur == "vertex":
                n = int(p[2])
        elif p[:1] == ["property"] and cur == "vertex":
            props.append((p[-1], "<" + T[p[1]]))
    return np.frombuffer(raw, dtype=np.dtype(props), count=n, offset=end), props

def write_ply(path, data, props):
    head = "ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % len(data)
    rev = {v: k for k, v in T.items() if k in ("float", "double", "uchar", "char", "short", "ushort", "int", "uint")}
    head += "".join(f"property {rev[t[1:]]} {name}\n" for name, t in props) + "end_header\n"
    with open(path, "wb") as f:
        f.write(head.encode("ascii"))
        f.write(np.ascontiguousarray(data).tobytes())

def load_mask(path):
    try:
        import torch
        m = torch.load(path, map_location="cpu")
        if isinstance(m, dict) and "mask" in m:
            m = m["mask"]
        return m.bool().numpy().reshape(-1)
    except ImportError:   # sans PyTorch : un masque bool est stocke tel quel dans l'archive .pt
        z = zipfile.ZipFile(path)
        data = [n for n in z.namelist() if re.search(r"/data/\d+$", n)]
        return np.frombuffer(z.read(data[0]), dtype=np.uint8) != 0

def safe(name):
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_") or "objet"

data, props = read_ply(scene)
N = len(data)
XYZ = np.stack([data["x"], data["y"], data["z"]], 1).astype(np.float64)
VOXEL = float((np.percentile(XYZ, 99, 0) - np.percentile(XYZ, 1, 0)).max()) / 150

def main_cluster(m):
    # SAGA attrape souvent des morceaux ailleurs dans la piece qui "ressemblent" a l'objet :
    # on garde le plus gros bloc d'un seul tenant (grille de voxels, ~1/150 de la scene)
    if os.environ.get("GS_OBJ_MAIN_ONLY", "1") != "1":
        return m
    try:
        from scipy import ndimage
    except ImportError:
        return m
    idx = np.flatnonzero(m)
    ijk = np.floor((XYZ[idx] - XYZ[idx].min(0)) / VOXEL).astype(np.int64)
    grid = np.zeros(ijk.max(0) + 1, bool)
    grid[tuple(ijk.T)] = True
    lab, n = ndimage.label(ndimage.binary_dilation(grid), structure=np.ones((3, 3, 3)))
    if n <= 1:
        return m
    pl = lab[tuple(ijk.T)]
    keep = np.zeros_like(m)
    keep[idx[pl == np.bincount(pl)[1:].argmax() + 1]] = True
    return keep

objects, report = [], []
for pt in sorted(glob.glob(os.path.join(seg_dir, "*.pt"))):
    name = os.path.splitext(os.path.basename(pt))[0]
    m = load_mask(pt)
    pct = m.mean() * 100 if len(m) else 0
    if len(m) != N:
        report.append(f"  ✗ {name} : masque d'un autre modele ({len(m)} != {N}), ignore")
    elif m.sum() == 0:
        report.append(f"  ✗ {name} : vide, ignore")
    elif pct > 95:
        report.append(f"  ✗ {name} : {pct:.0f} % de la scene = pas un objet (trop de clics ou save as sans segment3d), ignore")
    else:
        before = int(m.sum())
        m = main_cluster(m)
        objects.append((name, m))
        extra = f", {before - int(m.sum())} parasites hors du bloc principal retires" if before != int(m.sum()) else ""
        report.append(f"  ✓ {name} : {int(m.sum())} gaussiennes ({m.mean() * 100:.1f} %{extra})")
print("Masques SAGA trouves :")
print("\n".join(report) if report else "  (aucun)")
if not objects:
    sys.exit("[ERROR] Aucun objet valide : rien a preparer. Refaites la decoupe dans SAGA (1 a 3 clics droits par objet, segment3d, verifier le %, save as).")

if os.path.isdir(out):
    shutil.rmtree(out)
os.makedirs(os.path.join(out, "objets"))
union = np.zeros(N, bool)
lines = []
for name, m in objects:
    f = safe(name) + ".ply"
    write_ply(os.path.join(out, "objets", f), data[m], props)
    if (union & m).any():
        print(f"[WARN] {name} chevauche un objet precedent ({int((union & m).sum())} gaussiennes en commun).")
    union |= m
    lines.append(f"objets/{f} : {int(m.sum())} gaussiennes")
write_ply(os.path.join(out, "scene_seule.ply"), data[~union], props)
shutil.copyfile(scene, os.path.join(out, "scene_complete.ply"))
summary = ["Gaussian App - dossier final", "",
           f"scene_complete.ply : {N} gaussiennes (modele d'origine)",
           f"scene_seule.ply    : {int((~union).sum())} gaussiennes (scene sans les objets)", ""] + lines
open(os.path.join(out, "LISEZMOI.txt"), "w", encoding="utf-8").write("\n".join(summary) + "\n")
print("\n".join(summary))
print(f"[OK] Dossier pret : {out}")
PY
}

cmd_export() {
    # export <projet> [nom_masque] [--invert]
    local P="" m="" invert=0
    while [ $# -gt 0 ]; do
        case "$1" in
            --invert) invert=1 ;;
            *) if [ -z "$P" ]; then P="$1"; else m="$1"; fi ;;
        esac
        shift
    done
    require_project "$P"
    local S it pt n=0 suffix=""
    S="$(proj_path "$P")"; it="$(scene_iter "$S")"
    saga_activate || ensure_saga
    [ "$invert" = 1 ] && suffix="_SANS"
    for pt in "$S/model/segmentation_res"/*.pt; do
        [ -f "$pt" ] || continue
        [ -n "$m" ] && [ "$(basename "$pt" .pt)" != "$m" ] && continue
        export_mask "$S/model/point_cloud/iteration_$it/scene_point_cloud.ply" "$pt" \
            "$OUT_DIR/SEG_${P}_$(basename "$pt" .pt)${suffix}.ply" "$invert" && n=$((n + 1))
    done
    [ "$n" -gt 0 ] || warn "Aucun masque SAGA enregistre pour $P (bouton 'save as' dans la fenetre SAGA)."
}

print_saga_guide() {
    cat <<'EOF'
======================================================================
✂️  DECOUPER UN OBJET AVEC SAGA - mode d'emploi
======================================================================
 1. Dans le panneau "Control" a droite, cocher  clickmode
 2. Faire un CLIC DROIT sur l'objet a garder dans la vue 3D
    -> l'objet s'allume (apercu 2D)
 3. Regler le curseur ScoreThres (0.7 a 0.9) : seul l'objet doit
    rester colore. Scale : petit = pieces fines, grand = objet entier
    (multi-clickmode = plusieurs clics pour plusieurs objets)
 4. Cliquer  segment3d   -> il ne reste que l'objet en 3D
    ('roll_back' = annuler, 'clear' = tout recommencer)
 5. Taper un nom dans la case sous "save as", puis cliquer  save as
 6. Fermer la fenetre : le PLY decoupe est cree automatiquement dans
    output_ply/SEG_<projet>_<nom>.ply
 Navigation : clic gauche glisse = tourner, molette = zoom,
              clic milieu glisse = deplacer
======================================================================
EOF
}

cmd_gui() {
    local P="$1"
    require_project "$P"
    local S; S="$(proj_path "$P")"
    if [ -z "$(feat_iter "$S")" ]; then
        info "Scene pas encore preparee pour SAGA : preparation automatique..."
        cmd_run "$P" --saga || return 1
    fi
    if [ -z "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]; then
        if [ -S /tmp/.X11-unix/X0 ]; then export DISPLAY=:0; else die "Aucun ecran : la fenetre SAGA doit s'ouvrir sur une session graphique."; fi
    fi
    ensure_saga
    print_saga_guide
    local marker="$STATE_DIR/gui_start_$$"
    touch "$marker"
    mkdir -p "$S/model/segmentation_res"
    # Lance depuis model/ : les masques sont ranges dans model/segmentation_res/
    ( cd "$S/model" && python "$SAGA_DIR/saga_gui.py" -m "$S/model" -s "$(scene_iter "$S")" -f "$(feat_iter "$S")" )
    local pt it n=0
    it="$(scene_iter "$S")"
    for pt in "$S/model/segmentation_res"/*.pt; do
        [ -f "$pt" ] && [ "$pt" -nt "$marker" ] || continue
        export_mask "$S/model/point_cloud/iteration_$it/scene_point_cloud.ply" "$pt" \
            "$OUT_DIR/SEG_${P}_$(basename "$pt" .pt).ply" && n=$((n + 1))
    done
    rm -f "$marker"
    if [ "$n" -gt 0 ]; then ok "$n objet(s) decoupe(s) exporte(s) dans $OUT_DIR"; else warn "Aucun objet enregistre (bouton 'save as')."; fi
}

cmd_build() {
    # Traitement par lot de toutes les videos de input_video/
    local force=false v P S
    [ "${1:-}" = "--force" ] && force=true
    title "🎬 Traitement par lot de $SRC_VIDEO_DIR"
    local total=0 success=0 failed=0
    while IFS= read -r v; do
        total=$((total + 1))
        P="$(safe_name "$(basename "${v%.*}")")"
        [ -d "$(proj_path "$(legacy_name "$(basename "${v%.*}")")")" ] && P="$(legacy_name "$(basename "${v%.*}")")"
        S="$(proj_path "$P")"
        if [ -f "$OUT_DIR/FINAL_${P}.ply" ] && [ "$force" = false ]; then
            info "$P : PLY final deja present (--force pour recalculer)."
            success=$((success + 1)); continue
        fi
        if [ ! -d "$S" ]; then
            mkdir -p "$S"; echo "$v" >"$S/.source_video"
        fi
        if cmd_run "$P"; then success=$((success + 1)); else failed=$((failed + 1)); fi
    done < <(find "$SRC_VIDEO_DIR" -maxdepth 1 -type f | grep -Ei "$VIDEO_EXT_RE" | sort)
    [ "$total" -gt 0 ] || { warn "Aucune video dans $SRC_VIDEO_DIR"; return 0; }
    title "🏁 Bilan : $success/$total reussie(s), $failed echec(s)"
    ls -lh "$OUT_DIR"/*.ply 2>/dev/null
    [ "$failed" -eq 0 ]
}

cmd_clean() {
    [ $# -ge 1 ] || die "Usage: clean <input.ply> [output.ply] [min_opacity_logit=-2.0] [max_scale=3.0]"
    local in="$1" out="${2:-${1%.ply}_clean.ply}"
    saga_activate 2>/dev/null || true
    python3 - "$in" "$out" "${3:--2.0}" "${4:-3.0}" <<'PY'
import os, sys
import numpy as np
from plyfile import PlyData, PlyElement
inp, out, min_op, max_scale = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
data = PlyData.read(inp)["vertex"].data
mask = np.ones(len(data), dtype=bool)
if "opacity" in data.dtype.names:
    mask &= data["opacity"] >= min_op
if all(f"scale_{i}" in data.dtype.names for i in range(3)):
    mask &= np.maximum(np.maximum(data["scale_0"], data["scale_1"]), data["scale_2"]) <= max_scale
PlyData([PlyElement.describe(data[mask], "vertex")], text=False).write(out)
print(f"[OK] {out} : {mask.sum()}/{len(data)} gaussiennes conservees ({mask.mean()*100:.1f}%)")
PY
}

project_summary() {
    local S="$1" s=""
    [ -n "$(project_video "$S")" ] && s+="video "
    s+="input:$(count_images "$S/input") "
    colmap_ready "$S" && s+="cameras✓ "
    [ -f "$S/.synthetic_views" ] && s+="(vues virtuelles) "
    [ -n "$(scene_iter "$S")" ] && s+="3DGS✓ "
    [ "$(count_files "$S/sam_masks" pt)" -gt 0 ] && s+="sam:$(count_files "$S/sam_masks" pt) "
    [ -n "$(feat_iter "$S")" ] && s+="SAGA✓ "
    echo "$s"
}

cmd_status() {
    title "📊 Etat"
    echo "Racine : $BASE_DIR"
    show_gpu
    local t
    use_tools_env 2>/dev/null
    for t in ffmpeg colmap git conda; do
        printf "  %-8s %s\n" "$t" "$(command -v "$t" || echo '-- sera installe automatiquement --')"
    done
    printf "  %-8s %s\n" "SAGA" "$([ -f "$STATE_DIR/saga_ok" ] && echo 'pret' || echo '-- sera installe automatiquement --')"
    echo "--- Projets ($WORK_ROOT) ---"
    local d
    for d in "$WORK_ROOT"/*/; do
        d="${d%/}"
        case "$(basename "$d")" in images|sparse) continue ;; esac
        [ -d "$d" ] && printf "  %-45s %s\n" "$(basename "$d")" "$(project_summary "$d")"
    done
    echo "--- PLY ($OUT_DIR) ---"
    ls -lh "$OUT_DIR"/*.ply 2>/dev/null || echo "  Aucun."
}

cmd_ui() {
    command -v python3 &>/dev/null || die "python3 introuvable : utilisez 'bash $(basename "$SELF_PATH") menu'."
    exec python3 "$SCRIPT_DIR/gaussian_app_ui.py" --app "$SELF_PATH" --base "$BASE_DIR" --port "$GS_UI_PORT" "$@"
}

cmd_install_agy() {
    title "🛰️  Installation d'Antigravity CLI (sans droits root)"
    curl -fsSL https://antigravity.google/cli/install.sh | bash || return 1
    echo 'Ajoutez a votre ~/.bashrc : export PATH="$HOME/.antigravity/bin:$PATH"'
}

cmd_install_launcher() {
    local dir="$HOME/.local/share/applications"
    mkdir -p "$dir"
    cat >"$dir/gaussian-app.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=$APP_NAME
Comment=Video / photos / PLY -> Gaussian Splatting + decoupe SAGA
Exec=bash "$SELF_PATH" ui
Terminal=true
Icon=applications-graphics
Categories=Graphics;3DGraphics;
EOF
    ok "Raccourci cree dans le menu des applications ($dir/gaussian-app.desktop)."
    info "Si la cle USB change de point de montage, relancez cette commande."
}

cmd_help() {
    cat <<EOF
$APP_NAME - Gaussian Splatting + decoupe SAGA, tout-en-un

  bash $(basename "$SELF_PATH")                 Interface web (recommande)
  bash $(basename "$SELF_PATH") menu            Menu dans le terminal

  install                       Installe tout ce qui manque (sans root)
  import <chemin> [--name N]    Video, dossier de photos, projet COLMAP
                                (images/ + sparse/), archive .zip ou .ply
  run <projet>                  -> output_ply/FINAL_<projet>.ply
  run <projet> --saga           + preparation SAGA (SAM, echelles, features)
  gui <projet>                  Ouvre la decoupe SAGA (prepare si besoin)
  export <projet> [masque] [--invert]
                                Reexporte les objets decoupes en PLY
  package <projet>              output_ply/<projet>/ : scene_complete, scene_seule
                                (sans les objets) et objets/<nom>.ply
  reset <projet> <etape>        Efface les calculs depuis : frames | colmap | train | saga
                                (pour refaire avec d'autres reglages)
  settings                      Affiche les reglages actifs
  build [--force]               Traite toutes les videos de input_video/
  clean <in.ply> [out.ply] [opacite=-2] [echelle=3]
  status                        Etat des outils et des projets
  install-launcher              Raccourci dans le menu des applications
  install-agy                   Installe Antigravity CLI

Reglages : GS_FPS=$GS_FPS GS_MAX_FRAMES=$GS_MAX_FRAMES GS_ITERS=$GS_ITERS GS_SH_DEGREE=$GS_SH_DEGREE
           GS_MAX_IMAGE_SIZE=$GS_MAX_IMAGE_SIZE GS_SAM_DOWNSAMPLE=$GS_SAM_DOWNSAMPLE GS_SAGA_ITERS=$GS_SAGA_ITERS
           GS_SYNTH_VIEWS=$GS_SYNTH_VIEWS GS_YOLO=$GS_YOLO GS_YOLO_CLASSES=$GS_YOLO_CLASSES
           GS_360=$GS_360 GS_360_MAX_IMAGES=$GS_360_MAX_IMAGES GS_360_SIZE=$GS_360_SIZE GS_360_FOV=$GS_360_FOV GS_360_POLES=$GS_360_POLES
           (modifiables dans l'interface, enregistres dans $SETTINGS_FILE)
Log      : $MASTER_LOG
EOF
}

# ------------------------------------------------------------------------------
# Menu terminal (whiptail si disponible, sinon texte)
# ------------------------------------------------------------------------------
MENU_ITEMS=(
    ui       "Ouvrir l'interface web (recommande)"
    import   "Importer video / photos / COLMAP / PLY"
    run      "Reconstruire un projet -> PLY"
    gui      "Decouper un objet avec SAGA"
    build    "Traiter toutes les videos de input_video/"
    install  "Installer / reparer tout"
    status   "Etat des outils et projets"
    quit     "Quitter"
)

ask() {
    local answer
    if command -v whiptail &>/dev/null; then
        answer=$(whiptail --title "$APP_NAME" --inputbox "$1" 10 78 "$2" 3>&1 1>&2 2>&3) || return 1
    else
        read -r -p "$1 [$2] : " answer </dev/tty || return 1
        answer="${answer:-$2}"
    fi
    echo "$answer"
}

pick_project() {
    local names=() d
    for d in "$WORK_ROOT"/*/; do
        d="$(basename "$d")"
        case "$d" in images|sparse|'*') continue ;; esac
        names+=("$d")
    done
    [ ${#names[@]} -gt 0 ] || { echo ""; return 1; }
    if command -v whiptail &>/dev/null; then
        local items=() n
        for n in "${names[@]}"; do items+=("$n" ""); done
        whiptail --title "$APP_NAME" --menu "Projet :" 20 78 12 "${items[@]}" 3>&1 1>&2 2>&3
    else
        local i
        for i in "${!names[@]}"; do echo "  $((i + 1))) ${names[$i]}" >&2; done
        read -r -p "Projet : " i </dev/tty
        echo "${names[$((i - 1))]}"
    fi
}

run_sub() {
    bash "$SELF_PATH" "$@"
    echo ""
    read -r -p "Entree pour revenir au menu..." _ </dev/tty
}

cmd_menu() {
    local choice p
    while true; do
        if command -v whiptail &>/dev/null; then
            choice=$(whiptail --title "$APP_NAME" --menu "Racine : $BASE_DIR" 20 78 10 "${MENU_ITEMS[@]}" 3>&1 1>&2 2>&3) || return 0
        else
            echo ""; title "$APP_NAME - $BASE_DIR"
            local i n=1 keys=()
            for ((i = 0; i < ${#MENU_ITEMS[@]}; i += 2)); do
                printf "  %d) %s\n" "$n" "${MENU_ITEMS[$((i + 1))]}"
                keys+=("${MENU_ITEMS[$i]}"); n=$((n + 1))
            done
            read -r -p "Choix : " n </dev/tty || return 0
            [[ "$n" =~ ^[0-9]+$ ]] && [ "$n" -ge 1 ] && [ "$n" -le ${#keys[@]} ] || continue
            choice="${keys[$((n - 1))]}"
        fi
        case "$choice" in
            quit) return 0 ;;
            ui) cmd_ui ;;
            import) p=$(ask "Chemin (video, dossier, .zip ou .ply) :" "$HOME/") && run_sub import "$p" ;;
            run|gui) p=$(pick_project) && [ -n "$p" ] && run_sub "$choice" "$p" ;;
            *) run_sub "$choice" ;;
        esac
    done
}

# ------------------------------------------------------------------------------
# Point d'entree
# ------------------------------------------------------------------------------
main() {
    local cmd="${1:-ui}"
    [ $# -gt 0 ] && shift
    case "$cmd" in
        ui)               cmd_ui "$@" ;;
        menu)             cmd_menu ;;
        help|-h|--help)   cmd_help ;;
        install|setup)    start_logging "install"; cmd_install ;;
        import)           start_logging "import"; cmd_import "$@" ;;
        run)              start_logging "run $*"; cmd_run "$@" ;;
        gui|saga-gui)     start_logging "gui $*"; cmd_gui "$@" ;;
        export)           start_logging "export $*"; cmd_export "$@" ;;
        reset)            start_logging "reset $*"; cmd_reset "$@" ;;
        package)          start_logging "package $*"; cmd_package "$@" ;;
        settings)         set | grep -E '^GS_' ;;
        build)            start_logging "build"; cmd_build "$@" ;;
        clean)            start_logging "clean"; cmd_clean "$@" ;;
        status)           cmd_status ;;
        install-agy)      cmd_install_agy ;;
        install-launcher) cmd_install_launcher ;;
        *) err "Commande inconnue : $cmd"; cmd_help; exit 1 ;;
    esac
}

main "$@"
