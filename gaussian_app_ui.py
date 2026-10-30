#!/usr/bin/env python3
# ==============================================================================
# GAUSSIAN APP - Interface web locale (bibliotheque standard Python uniquement)
# Lancee par : bash gaussian_app.sh   (ou bash gaussian_app.sh ui)
# Chaque action execute "bash gaussian_app.sh <commande>" et affiche son log.
# ==============================================================================

import argparse
import codecs
import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")
IMG_RE = re.compile(r"\.(jpe?g|png)$", re.I)
VIDEO_RE = re.compile(r"\.(mp4|mov|mkv|avi|webm|m4v|insv)$", re.I)
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
MAX_LINES = 5000

APP = BASE = WORK = OUT = STATE = ""
GPU = ""

# Reglages modifiables dans l'interface -> .gaussian_app/settings.env (lu par gaussian_app.sh)
SETTINGS = [
    {"key": "GS_FPS", "group": "Vidéo", "label": "Images extraites par seconde", "type": "float", "min": 0.2, "max": 10, "default": 2,
     "help": "Plus d'images = meilleur recouvrement mais calcul plus long. 1 à 3 conseillé ; montez si vous filmez vite."},
    {"key": "GS_MAX_FRAMES", "group": "Vidéo", "label": "Nombre maximum d'images", "type": "int", "min": 20, "max": 1500, "default": 250,
     "help": "Au-delà, les images sont réparties régulièrement sur la vidéo."},
    {"key": "GS_360", "group": "Vidéo 360°", "label": "Mode 360°", "type": "choice", "choices": ["auto", "on", "off"], "default": "auto",
     "help": "auto = .insv brut (2 fisheye → 3 vues par objectif) et vidéos 2:1 exportées en équirectangulaire (→ 6 vues tous les 60°). La caméra peut être retournée : l'horizon reste horizontal."},
    {"key": "GS_360_MAX_IMAGES", "group": "Vidéo 360°", "label": "Nombre maximum de vues découpées", "type": "int", "min": 60, "max": 2000, "default": 480,
     "help": "Le FPS est réduit automatiquement pour ne pas dépasser ce total. 480 vues en 1280 px ≈ 9 Go de RAM à l'entraînement."},
    {"key": "GS_360_SIZE", "group": "Vidéo 360°", "label": "Taille des vues découpées (pixels)", "type": "int", "min": 512, "max": 2048, "default": 1280,
     "help": "Votre README utilisait 1600 : plus net, mais 1,6× plus de RAM."},
    {"key": "GS_360_FOV", "group": "Vidéo 360°", "label": "Champ de chaque vue (degrés)", "type": "int", "min": 70, "max": 120, "default": 100,
     "help": "100° = bon recouvrement entre vues voisines (équirectangulaire)."},
    {"key": "GS_360_POLES", "group": "Vidéo 360°", "label": "Ajouter une vue vers le haut et une vers le bas", "type": "bool", "default": 0,
     "help": "Plafond et sol vus de face. Inutile si vous faites des tours à plusieurs hauteurs."},
    {"key": "GS_MAX_IMAGE_SIZE", "group": "Reconstruction", "label": "Résolution maximale (pixels)", "type": "choice", "choices": ["auto", 1024, 1280, 1600, 2000, 2400, 3200], "default": "auto",
     "help": "Pour COLMAP et l'entraînement. auto = selon la carte graphique et la RAM (1600 sur une RTX 2080) ; réduit tout seul si la mémoire déborde."},
    {"key": "GS_ITERS", "group": "Reconstruction", "label": "Itérations d'entraînement 3DGS", "type": "int", "min": 1000, "max": 60000, "default": 30000,
     "help": "7000 = essai rapide (~5 min), 30000 = qualité (~30-40 min sur RTX 2080)."},
    {"key": "GS_SH_DEGREE", "group": "Reconstruction", "label": "Reflets (degré SH)", "type": "choice", "choices": [0, 1, 2, 3], "default": 1,
     "help": "1 = fichier léger, idéal Meta Quest 3. 3 = reflets complets, fichier 3× plus gros."},
    {"key": "GS_SAM_DOWNSAMPLE", "group": "SAGA", "label": "Réduction des images pour SAM", "type": "choice", "choices": ["auto", 1, 2, 4, 8], "default": "auto",
     "help": "auto = selon la mémoire de la carte graphique et la taille des images. Si la carte déborde, l'app réessaie seule en plus petit."},
    {"key": "GS_SAGA_RES", "group": "SAGA", "label": "Largeur des images pour SAGA", "type": "choice", "choices": ["auto", 640, 800, 1024, 1280, 1600], "default": "auto",
     "help": "get_scale et features gardent toutes les vues en RAM. auto = selon la RAM libre et le nombre d'images ; réduit tout seul si ça déborde."},
    {"key": "GS_SAGA_ITERS", "group": "SAGA", "label": "Itérations SAGA", "type": "int", "min": 1000, "max": 30000, "default": 10000,
     "help": "Apprentissage des features de découpe."},
    {"key": "GS_SYNTH_VIEWS", "group": "SAGA", "label": "Vues virtuelles (PLY seul)", "type": "int", "min": 50, "max": 1000, "default": 300,
     "help": "~2 s de SAM et ~20 Mo de RAM par vue."},
    {"key": "GS_YOLO", "group": "YOLO (optionnel)", "label": "Masquer la personne qui filme avec YOLO", "type": "bool", "default": 0,
     "help": "Pas nécessaire pour SAGA. Si activé : silhouette détectée (double passage endroit + caméra retournée), ignorée par COLMAP ET par l'entraînement (comme « Mask Mode : Ignore » de LichtFeld). Aperçus rouges consultables sur le projet."},
    {"key": "GS_YOLO_CLASSES", "group": "YOLO (optionnel)", "label": "Classes à masquer", "type": "text", "default": "person",
     "help": "Noms COCO en anglais séparés par des virgules : person, car, bicycle, dog, cat…"},
]


def settings_path():
    return os.path.join(STATE, "settings.env")


def read_settings():
    vals = {s["key"]: s["default"] for s in SETTINGS}
    try:
        lines = open(settings_path()).read().splitlines()
    except OSError:
        return vals
    v2 = "GS_SETTINGS_VERSION=2" in lines
    for line in lines:
        k, _, v = line.strip().partition("=")
        if k in vals:
            # anciens reglages : les valeurs par defaut d'avant deviennent "auto" (comme gaussian_app.sh)
            if not v2 and (k, v) in (("GS_MAX_IMAGE_SIZE", "1600"), ("GS_SAM_DOWNSAMPLE", "2")):
                v = "auto"
            vals[k] = v
    return vals


def hardware():
    # materiel detecte par gaussian_app.sh (valeurs utilisees par les reglages "auto")
    try:
        out = subprocess.run(["bash", APP, "hw"], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    return dict(l.split("=", 1) for l in out.splitlines() if "=" in l)


def validate_settings(data):
    out = {}
    for s in SETTINGS:
        k, v = s["key"], data.get(s["key"], s["default"])
        try:
            if s["type"] in ("int", "float"):
                num = int(float(v)) if s["type"] == "int" else round(float(v), 3)
                if not s["min"] <= num <= s["max"]:
                    raise ValueError(f"entre {s['min']} et {s['max']}")
                out[k] = num
            elif s["type"] == "choice":
                match = [c for c in s["choices"] if str(c) == str(v)]
                if not match:
                    raise ValueError(f"parmi {s['choices']}")
                out[k] = match[0]
            elif s["type"] == "bool":
                out[k] = 1 if v in (1, True, "1", "true", "on") else 0
            else:
                v = str(v).strip()
                if not re.fullmatch(r"[A-Za-z ,_-]{1,200}", v):
                    raise ValueError("lettres et virgules uniquement")
                out[k] = v
        except (TypeError, ValueError) as e:
            raise ValueError(f"{s['label']} : {e}")
    return out


# ------------------------------------------------------------------------------
# Execution des commandes (une seule a la fois : le GPU est partage)
# ------------------------------------------------------------------------------
class Job:
    def __init__(self):
        self.lock = threading.Lock()
        self.proc = None
        self.lines = []
        self.offset = 0
        self.progress = ""
        self.title = ""
        self.rc = None
        self.queue = []
        self.on_done = None

    @property
    def running(self):
        return self.proc is not None and self.proc.poll() is None

    def start(self, title, args, queue=None, on_done=None):
        with self.lock:
            if self.running:
                return False
            self.lines, self.offset, self.progress = [], 0, ""
            self._launch(title, args)
            self.queue = list(queue or [])
            self.on_done = on_done
        return True

    def _launch(self, title, args):
        self.title, self.rc = title, None
        self._push(f"$ bash gaussian_app.sh {' '.join(args)}")
        env = dict(os.environ, PYTHONUNBUFFERED="1", GA_FROM_UI="1")
        self.proc = subprocess.Popen([os.environ.get("GA_BASH", "bash"), APP] + args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True, env=env)
        threading.Thread(target=self._reader, args=(self.proc,), daemon=True).start()

    def _push(self, line):
        self.lines.append(ANSI_RE.sub("", line))
        if len(self.lines) > MAX_LINES:
            drop = len(self.lines) - MAX_LINES
            del self.lines[:drop]
            self.offset += drop

    def _reader(self, proc):
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        partial, after_cr = "", False
        fd = proc.stdout.fileno()
        while True:
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            text = decoder.decode(chunk)
            pos = 0
            # \r = barre de progression (tqdm, pip...) : on remplace la ligne courante
            for m in re.finditer(r"\r\n|\n|\r", text):
                partial += text[pos:m.start()]
                if m.group() == "\r":
                    self.progress = ANSI_RE.sub("", partial)
                    after_cr = True
                else:
                    if partial or not after_cr:
                        self._push(partial)
                    self.progress, after_cr = "", False
                partial = ""
                pos = m.end()
            partial += text[pos:]
        if partial:
            self._push(partial)
        rc = proc.wait()
        with self.lock:
            self.rc = rc
            self.progress = ""
            self._push(f"--- termine ({'OK' if rc == 0 else 'ERREUR code ' + str(rc)}) ---")
            if rc == 0 and self.on_done:
                try:
                    self.queue += self.on_done(self.lines) or []
                except Exception as e:  # ne doit jamais bloquer l'enchainement
                    self._push(f"[ui] {e}")
                self.on_done = None
            if rc == 0 and self.queue:
                title, args = self.queue.pop(0)
                self._push("")
                self._launch(title, args)
            else:
                self.queue = []

    def cancel(self):
        with self.lock:
            self.queue = []
            if self.running:
                try:
                    os.killpg(self.proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                self._push("--- annulation demandee ---")


JOB = Job()


# ------------------------------------------------------------------------------
# Etat des projets (miroir des fichiers ecrits par gaussian_app.sh)
# ------------------------------------------------------------------------------
def count(d, rx=IMG_RE):
    try:
        return sum(1 for f in os.listdir(d) if rx.search(f))
    except OSError:
        return 0


def iterations(S, fname):
    root = os.path.join(S, "model", "point_cloud")
    its = []
    try:
        for d in os.listdir(root):
            if os.path.isfile(os.path.join(root, d, fname)):
                its.append(int(d.split("_")[-1]))
    except (OSError, ValueError):
        pass
    return max(its) if its else None


def file_info(path):
    st = os.stat(path)
    return {"name": os.path.basename(path), "size": st.st_size, "mtime": st.st_mtime}


def project_info(name):
    S = os.path.join(WORK, name)
    video = ""
    sv = os.path.join(S, ".source_video")
    if os.path.isfile(sv):
        video = open(sv).read().strip()
    elif os.path.isdir(os.path.join(S, "source")):
        video = next((f for f in os.listdir(os.path.join(S, "source")) if VIDEO_RE.search(f)), "")
    n_input, n_images = count(os.path.join(S, "input")), count(os.path.join(S, "images"))
    imported_ply = os.path.isfile(os.path.join(S, ".imported_ply"))
    synthetic = os.path.isfile(os.path.join(S, ".synthetic_views"))
    legacy = os.path.isdir(os.path.join(S, "colmap", "sparse", "0"))
    cameras = os.path.isdir(os.path.join(S, "sparse", "0")) and n_images >= 3 and not legacy_unconverted(S)
    if video:
        kind = "vidéo 360°" if os.path.isfile(os.path.join(S, ".is360")) else "video"
    elif imported_ply and not n_input and not (cameras and not synthetic):
        kind = "ply"
    elif os.path.isfile(os.path.join(S, ".unordered")):
        kind = "photos"
    elif legacy:
        kind = "ancien projet"
    else:
        kind = "colmap"
    seg_dir = os.path.join(S, "model", "segmentation_res")
    masks = sorted(f[:-3] for f in os.listdir(seg_dir) if f.endswith(".pt")) if os.path.isdir(seg_dir) else []
    outs = sorted(f for f in os.listdir(OUT) if f.startswith(f"SEG_{name}_") and f.endswith(".ply")) if os.path.isdir(OUT) else []
    final = f"FINAL_{name}.ply"
    return {
        "name": name, "kind": kind, "video": os.path.basename(video),
        "input": n_input, "images": n_images, "cameras": cameras, "synthetic": synthetic,
        "model": iterations(S, "scene_point_cloud.ply"), "features": iterations(S, "contrastive_feature_point_cloud.ply"),
        "sam": count(os.path.join(S, "sam_masks"), re.compile(r"\.pt$")),
        "masks": masks, "segs": outs,
        "final": final if os.path.isfile(os.path.join(OUT, final)) else "",
        "imported_ply": imported_ply,
        "imported_colmap": os.path.isfile(os.path.join(S, ".imported_colmap")),
        "legacy": legacy or os.path.isdir(os.path.join(S, "_legacy_images")),
        "yolo": count(os.path.join(S, "yolo_preview")),
        "package": package_files(name),
        "mtime": os.path.getmtime(S),
    }


def package_files(name):
    # fichiers du dossier final output_ply/<projet>/ (chemins relatifs a output_ply)
    out = []
    for sub in ("", "objets"):
        d = os.path.join(OUT, name, sub)
        if os.path.isdir(d):
            out += sorted(f"{name}/{sub + '/' if sub else ''}{f}" for f in os.listdir(d)
                          if os.path.isfile(os.path.join(d, f)) and f.lower().endswith((".ply", ".txt")))
    return out


def legacy_unconverted(S):
    return os.path.isdir(os.path.join(S, "colmap", "sparse", "0")) and not os.path.isdir(os.path.join(S, "distorted"))


def state():
    projects = []
    if os.path.isdir(WORK):
        for n in os.listdir(WORK):
            if n in ("images", "sparse") or not NAME_RE.match(n) or n.startswith(".") or not os.path.isdir(os.path.join(WORK, n)):
                continue
            try:
                projects.append(project_info(n))
            except OSError:
                pass
    projects.sort(key=lambda p: -p["mtime"])
    outputs = []
    if os.path.isdir(OUT):
        outputs = sorted((file_info(os.path.join(OUT, f)) for f in os.listdir(OUT) if f.lower().endswith(".ply")),
                         key=lambda f: -f["mtime"])
    return {
        "base": BASE, "gpu": GPU,
        "display": bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")),
        "installed": {"tools": os.path.isfile(os.path.join(STATE, "tools_ok")),
                      "saga": os.path.isfile(os.path.join(STATE, "saga_ok"))},
        "projects": projects, "outputs": outputs,
        "job": {"running": JOB.running, "title": JOB.title, "rc": JOB.rc, "queued": len(JOB.queue)},
    }


def projects_from_log(lines):
    return [l.split("=", 1)[1].strip() for l in lines if l.startswith("PROJECT=")]


# ------------------------------------------------------------------------------
# Serveur HTTP
# ------------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "GaussianApp/1.0"

    def log_message(self, *a):
        pass

    def send_json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def read_json(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/":
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif u.path == "/api/state":
            self.send_json(state())
        elif u.path == "/api/log":
            since = int(q.get("since", ["0"])[0])
            with JOB.lock:
                start = max(since, JOB.offset)
                lines = JOB.lines[start - JOB.offset:]
                self.send_json({"next": JOB.offset + len(JOB.lines), "lines": lines, "progress": JOB.progress,
                                "running": JOB.running, "rc": JOB.rc, "title": JOB.title, "reset": since < JOB.offset})
        elif u.path == "/api/settings":
            self.send_json({"spec": SETTINGS, "values": read_settings(), "hw": hardware()})
        elif u.path == "/api/browse":
            self.browse(q.get("path", [os.path.expanduser("~")])[0])
        elif u.path == "/api/previews":
            p = q.get("project", [""])[0]
            d = os.path.join(WORK, p, "yolo_preview")
            if not NAME_RE.match(p) or not os.path.isdir(d):
                return self.send_json({"files": []})
            self.send_json({"files": sorted(f for f in os.listdir(d) if IMG_RE.search(f))})
        elif u.path.startswith("/preview/"):
            parts = unquote(u.path[len("/preview/"):]).split("/")
            if len(parts) != 2 or not all(NAME_RE.match(x) for x in parts):
                return self.send_error(400)
            path = os.path.join(WORK, parts[0], "yolo_preview", parts[1])
            if not os.path.isfile(path):
                return self.send_error(404)
            data = open(path, "rb").read()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif u.path.startswith("/download/"):
            self.download(unquote(u.path[len("/download/"):]))
        else:
            self.send_error(404)

    def browse(self, path):
        path = os.path.abspath(os.path.expanduser(path or "~"))
        if not os.path.isdir(path):
            path = os.path.dirname(path)
        try:
            entries = []
            for e in sorted(os.scandir(path), key=lambda e: (not e.is_dir(), e.name.lower())):
                if e.name.startswith("."):
                    continue
                entries.append({"name": e.name, "dir": e.is_dir(), "size": e.stat().st_size if e.is_file() else 0})
            self.send_json({"path": path, "parent": os.path.dirname(path), "entries": entries,
                            "places": [os.path.expanduser("~"), BASE, "/media", "/mnt"]})
        except OSError as e:
            self.send_json({"error": str(e)}, 400)

    def download(self, name):
        # fichier de output_ply/ ou de son dossier final <projet>/(objets/), jamais au-dela
        root = os.path.realpath(OUT)
        path = os.path.realpath(os.path.join(OUT, *name.split("/")))
        if not path.startswith(root + os.sep) or not name.lower().endswith((".ply", ".txt")):
            return self.send_error(400)
        name = os.path.basename(path)
        if not os.path.isfile(path):
            return self.send_error(404)
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(os.path.getsize(path)))
        self.send_header("Content-Disposition", f'attachment; filename="{name}"')
        self.end_headers()
        with open(path, "rb") as f:
            shutil.copyfileobj(f, self.wfile, 1 << 20)

    def do_POST(self):
        u = urlparse(self.path)
        if u.path == "/api/upload":
            return self.upload(parse_qs(u.query))
        try:
            data = self.read_json()
        except ValueError:
            return self.send_json({"error": "JSON invalide"}, 400)
        if u.path == "/api/import":
            return self.do_import(data)
        if u.path == "/api/action":
            return self.action(data)
        if u.path == "/api/settings":
            try:
                vals = validate_settings(data)
            except ValueError as e:
                return self.send_json({"error": str(e)}, 400)
            with open(settings_path(), "w") as f:
                f.write("GS_SETTINGS_VERSION=2\n")
                f.writelines(f"{k}={v}\n" for k, v in vals.items())
            return self.send_json({"ok": True, "values": vals})
        if u.path == "/api/cancel":
            JOB.cancel()
            return self.send_json({"ok": True})
        self.send_error(404)

    def upload(self, q):
        token = q.get("token", [""])[0]
        rel = q.get("rel", [""])[0].replace("\\", "/")
        parts = [p for p in rel.split("/") if p not in ("", ".")]
        if not NAME_RE.match(token) or not parts or ".." in parts:
            return self.send_json({"error": "chemin invalide"}, 400)
        dest = os.path.join(STATE, "uploads", token, *parts)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        remaining = int(self.headers.get("Content-Length") or 0)
        with open(dest, "wb") as f:
            while remaining > 0:
                chunk = self.rfile.read(min(1 << 20, remaining))
                if not chunk:
                    break
                f.write(chunk)
                remaining -= len(chunk)
        if remaining:
            return self.send_json({"error": "envoi interrompu"}, 400)
        self.send_json({"ok": True})

    def do_import(self, data):
        token, path = data.get("token", ""), data.get("path", "")
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", data.get("name", "")).strip("_")
        then = data.get("then", "run")
        if token:
            if not NAME_RE.match(token):
                return self.send_json({"error": "token invalide"}, 400)
            src = os.path.join(STATE, "uploads", token)
            args = ["import", src, "--move"]
        elif path and os.path.exists(path):
            src, args = path, ["import", path]
        else:
            return self.send_json({"error": "Fichier ou dossier introuvable"}, 400)
        if name:
            args += ["--name", name]

        def after_import(lines):
            if token:
                shutil.rmtree(os.path.join(STATE, "uploads", token), ignore_errors=True)
            extra = {"run": [], "saga": ["--saga"], "gui": ["--gui"]}.get(then)
            if extra is None:
                return []
            return [(f"Projet {p}", ["run", p] + extra) for p in projects_from_log(lines)]

        if not JOB.start("Import", args, on_done=after_import):
            return self.send_json({"error": "Une tache est deja en cours"}, 409)
        self.send_json({"ok": True})

    def action(self, data):
        act, p = data.get("action", ""), data.get("project", "")
        if act not in ("install", "build") and (not NAME_RE.match(p) or not os.path.isdir(os.path.join(WORK, p))):
            return self.send_json({"error": "projet inconnu"}, 400)
        table = {
            "install": ("Installation", ["install"]),
            "build": ("Videos de input_video/", ["build"]),
            "run": (f"Creer le PLY - {p}", ["run", p]),
            "saga": (f"Preparer SAGA - {p}", ["run", p, "--saga"]),
            "gui": (f"Decoupe SAGA - {p}", ["gui", p]),
            "export": (f"Export des objets - {p}", ["export", p] + (["--invert"] if data.get("invert") else [])),
            "clean": (f"Nettoyage - {p}", ["clean", os.path.join(OUT, f"FINAL_{p}.ply")]),
            "package": (f"Dossier final - {p}", ["package", p]),
            "reset": (f"Refaire depuis '{data.get('step')}' - {p}", ["reset", p, str(data.get("step"))]),
        }
        if act == "reset" and data.get("step") not in ("frames", "colmap", "train", "saga"):
            return self.send_json({"error": "etape inconnue"}, 400)
        if act not in table:
            return self.send_json({"error": "action inconnue"}, 400)
        if not JOB.start(*table[act]):
            return self.send_json({"error": "Une tache est deja en cours"}, 409)
        self.send_json({"ok": True})


# ------------------------------------------------------------------------------
# Page web
# ------------------------------------------------------------------------------
PAGE = r"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gaussian App</title>
<style>
:root{--bg:#f4f5f7;--card:#fff;--ink:#1d2330;--mute:#6b7383;--line:#e3e6eb;--acc:#5b5bd6;--acc2:#ecebff;
--ok:#1f9d55;--warn:#c27c0e;--bad:#d14343;--con:#11141a;--conink:#d6dae2;--r:14px;color-scheme:light}
/* Sombre : choisi dans l'en-tete, ou automatique selon le systeme */
:root[data-theme=dark]{--bg:#0f1115;--card:#181b22;--ink:#e7e9ee;--mute:#9098a8;--line:#2a2f3a;
--acc:#8b8bf5;--acc2:#24264a;--ok:#3ccf7e;--warn:#e0a83a;--bad:#f06a6a;--con:#0a0c10;color-scheme:dark}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#0f1115;--card:#181b22;--ink:#e7e9ee;--mute:#9098a8;--line:#2a2f3a;
--acc:#8b8bf5;--acc2:#24264a;--ok:#3ccf7e;--warn:#e0a83a;--bad:#f06a6a;--con:#0a0c10;color-scheme:dark}}
*{box-sizing:border-box}body{margin:0;font:15px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;background:var(--bg);color:var(--ink)}
header{padding:22px 28px 10px;display:flex;flex-wrap:wrap;gap:14px;align-items:center;justify-content:space-between}
h1{margin:0;font-size:22px}h1 small{display:block;font-size:13px;font-weight:400;color:var(--mute);margin-top:2px}
.pills{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
.pill{font-size:12.5px;padding:5px 10px;border-radius:99px;background:var(--card);border:1px solid var(--line);color:var(--mute)}
.pill.ok{color:var(--ok)}.pill.no{color:var(--warn)}
main{display:grid;grid-template-columns:minmax(0,1fr) 440px;gap:20px;padding:10px 28px 28px}
@media (max-width:1050px){main{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:var(--r);padding:18px 20px;margin-bottom:18px}
h2{font-size:16px;margin:0 0 12px}
button,.btn{font:inherit;border:1px solid var(--line);background:var(--card);color:var(--ink);padding:8px 14px;border-radius:10px;cursor:pointer;text-decoration:none;display:inline-flex;gap:6px;align-items:center}
button:hover:not(:disabled),.btn:hover{border-color:var(--acc)}
button.primary{background:var(--acc);border-color:var(--acc);color:#fff;font-weight:600}
button:disabled{opacity:.45;cursor:not-allowed}
.drop{border:2px dashed var(--line);border-radius:var(--r);padding:28px 18px;text-align:center;transition:.15s}
.drop.over{border-color:var(--acc);background:var(--acc2)}
.drop b{font-size:17px}.drop p{margin:6px 0 14px;color:var(--mute)}
.row{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
.types{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px;margin:14px 0 4px}
.type{font-size:13px;border:1px solid var(--line);border-radius:10px;padding:8px 10px;color:var(--mute)}.type b{color:var(--ink);display:block}
input[type=text]{font:inherit;padding:8px 10px;border-radius:10px;border:1px solid var(--line);background:var(--bg);color:var(--ink);min-width:220px;flex:1}
label.opt{display:flex;gap:6px;align-items:center;font-size:14px;cursor:pointer}
.bar{height:8px;background:var(--line);border-radius:9px;overflow:hidden;margin-top:12px;display:none}.bar i{display:block;height:100%;width:0;background:var(--acc)}
.proj{border:1px solid var(--line);border-radius:12px;padding:14px 16px;margin-bottom:12px}
.proj h3{margin:0;font-size:15px;word-break:break-all}.kind{font-size:12px;color:var(--mute);margin-left:6px;font-weight:400}
.steps{display:flex;gap:4px;margin:12px 0}
.st{flex:1;font-size:11.5px;text-align:center;padding:6px 2px;border-radius:8px;background:var(--bg);color:var(--mute)}
.st.done{background:var(--acc2);color:var(--acc);font-weight:600}
.files{font-size:13px;margin-top:10px;display:flex;flex-direction:column;gap:4px}
.files a{color:var(--acc)}
.console{position:sticky;top:12px}
/* Console plein ecran */
.console.full{position:fixed;inset:0;z-index:20;background:var(--con);top:0}
.console.full .card{margin:0;height:100%;border:0;border-radius:0;background:var(--con);color:var(--conink);display:flex;flex-direction:column;padding:14px 18px}
.console.full .con{flex:1;height:auto;font-size:var(--confont,14px);border-radius:8px}
.console.full .status.run{color:#8b8bf5}.console.full .kind{color:#9098a8}
.console.full button{background:#181b22;color:#e7e9ee;border-color:#2a2f3a}
.conbtns{display:flex;gap:6px;align-items:center}
.onlyfull{display:none}.console.full .onlyfull{display:inline-flex}
.con{background:var(--con);color:var(--conink);border-radius:12px;padding:12px;font:12.5px/1.45 ui-monospace,Menlo,Consolas,monospace;height:58vh;overflow:auto;white-space:pre-wrap;word-break:break-word}
.prog{font:12.5px ui-monospace,Menlo,Consolas,monospace;color:var(--acc);min-height:18px;margin:8px 0;white-space:pre;overflow:hidden;text-overflow:ellipsis}
.status{font-weight:600}.status.run{color:var(--acc)}.status.ok{color:var(--ok)}.status.bad{color:var(--bad)}
.empty{color:var(--mute);font-size:14px}
.modal{position:fixed;inset:0;background:#0008;display:none;align-items:center;justify-content:center;padding:16px;z-index:9}
.modal.show{display:flex}.box{background:var(--card);border-radius:var(--r);max-width:620px;width:100%;max-height:88vh;overflow:auto;padding:22px}
.guide li{margin:8px 0}.guide kbd{background:var(--acc2);color:var(--acc);border-radius:6px;padding:1px 6px;font-weight:600}
.bl{list-style:none;padding:0;margin:10px 0;max-height:46vh;overflow:auto;border:1px solid var(--line);border-radius:10px}
.bl li{padding:7px 12px;border-bottom:1px solid var(--line);cursor:pointer;display:flex;justify-content:space-between;gap:10px}
.bl li:hover{background:var(--acc2)}.bl small{color:var(--mute)}
.note{font-size:13px;color:var(--mute);margin-top:10px}
summary{cursor:pointer;list-style:none}summary::-webkit-details-marker{display:none}
summary h2::before{content:"▸ ";color:var(--mute)}details[open] summary h2::before{content:"▾ "}
.sgroup{margin-top:16px}.sgroup h4{margin:0 0 8px;font-size:13px;color:var(--acc);text-transform:uppercase;letter-spacing:.04em}
.field{display:grid;grid-template-columns:minmax(0,1fr) 150px;gap:4px 14px;align-items:center;padding:8px 0;border-top:1px solid var(--line)}
.field label{font-size:14px}.field small{grid-column:1;color:var(--mute);font-size:12.5px}
.field input,.field select{font:inherit;padding:6px 8px;border-radius:8px;border:1px solid var(--line);background:var(--bg);color:var(--ink);width:100%}
.field input[type=checkbox]{width:20px;height:20px;justify-self:end}
.field.changed label{color:var(--acc);font-weight:600}
select.redo{font:inherit;padding:7px 8px;border-radius:10px;border:1px solid var(--line);background:var(--card);color:var(--ink)}
</style>
<script>
// Theme applique avant l'affichage (pas de flash blanc)
try { const t = localStorage.getItem('ga_theme'); if (t === 'dark' || t === 'light') document.documentElement.dataset.theme = t; } catch (e) {}
</script>
</head>
<body>
<header>
  <h1>Gaussian App<small>Vidéo · photos · COLMAP · PLY → Gaussian Splatting → découpe d'objets SAGA</small></h1>
  <div class="row">
    <div class="pills" id="pills"></div>
    <button id="themeBtn" onclick="cycleTheme()" title="Thème : automatique, sombre ou clair"></button>
  </div>
</header>
<main>
<div>
  <section class="card">
    <h2>1. Ajouter une scène</h2>
    <div class="drop" id="drop">
      <b>Glissez ici vos fichiers ou un dossier</b>
      <p>Tout est détecté automatiquement</p>
      <div class="row" style="justify-content:center">
        <button class="primary" onclick="pick(false)">Choisir des fichiers</button>
        <button onclick="pick(true)">Choisir un dossier</button>
        <button onclick="openBrowse()" title="Pour les gros fichiers déjà sur la machine ou la clé : aucune copie">Fichier déjà sur la machine…</button>
      </div>
      <input type="file" id="fFiles" multiple hidden>
      <input type="file" id="fDir" webkitdirectory hidden>
    </div>
    <div class="types">
      <div class="type"><b>🎬 Vidéo</b>.mp4 .mov .mkv…</div>
      <div class="type"><b>🖼️ Photos</b>déjà extraites (jpg/png)</div>
      <div class="type"><b>📐 COLMAP</b>dossier images/ + sparse/</div>
      <div class="type"><b>✨ PLY</b>Gaussian Splatting (même compressé SuperSplat), direct pour SAGA</div>
    </div>
    <div class="row" style="margin-top:12px">
      <input type="text" id="pname" placeholder="Nom du projet (automatique)">
    </div>
    <div class="row" style="margin-top:12px">
      <span style="font-size:14px;color:var(--mute)">Ensuite :</span>
      <label class="opt"><input type="radio" name="then" value="none"> rien</label>
      <label class="opt"><input type="radio" name="then" value="run" checked> créer le PLY</label>
      <label class="opt"><input type="radio" name="then" value="saga"> PLY + préparer SAGA</label>
      <label class="opt"><input type="radio" name="then" value="gui"> tout + ouvrir la découpe</label>
    </div>
    <div class="bar" id="bar"><i></i></div>
    <div class="note" id="upnote"></div>
  </section>

  <details class="card" id="settingsCard">
    <summary><h2 style="display:inline">⚙️ Réglages</h2> <span class="kind" id="setSummary"></span></summary>
    <div class="row" style="margin:14px 0 4px">
      <span style="font-size:14px;color:var(--mute)">Préréglages :</span>
      <button onclick="preset('fast')" title="Pour vérifier qu'une vidéo marche">⚡ Essai rapide</button>
      <button onclick="preset('normal')">Normal</button>
      <button onclick="preset('quality')" title="Plus d'images, entraînement complet">💎 Qualité</button>
    </div>
    <div class="note" id="hwNote" style="margin:6px 0 10px"></div>
    <div id="setForm"></div>
    <div class="row" style="justify-content:space-between;margin-top:14px">
      <span class="note" style="margin:0">S'applique aux étapes pas encore calculées. Pour refaire un projet existant : « Refaire depuis… » sur le projet.</span>
      <button class="primary" onclick="saveSettings()">Enregistrer</button>
    </div>
  </details>

  <section class="card">
    <div class="row" style="justify-content:space-between;margin-bottom:6px">
      <h2 style="margin:0">2. Vos projets</h2>
      <button onclick="act('build')" title="Traite toutes les vidéos posées dans input_video/">Traiter input_video/</button>
    </div>
    <div id="projects"></div>
  </section>

  <section class="card">
    <h2>Fichiers PLY produits <span class="kind" id="outdir"></span></h2>
    <div id="outputs" class="files"></div>
  </section>
</div>

<div class="console" id="console">
  <section class="card">
    <div class="row" style="justify-content:space-between">
      <div><div class="status" id="jstatus">Aucune tâche</div><div class="kind" id="jtitle" style="margin:0"></div></div>
      <div class="conbtns">
        <button class="onlyfull" onclick="conFont(-1)" title="Texte plus petit">A−</button>
        <button class="onlyfull" onclick="conFont(1)" title="Texte plus grand">A+</button>
        <button id="cancel" onclick="cancelJob()" disabled>Annuler</button>
        <button id="fullBtn" onclick="toggleFull()" title="Console en plein écran (Échap pour sortir)">⛶ Plein écran</button>
      </div>
    </div>
    <div class="prog" id="prog"></div>
    <div class="con" id="con"></div>
  </section>
</div>
</main>

<div class="modal" id="guide"><div class="box">
  <h2>✂️ Découper un objet avec SAGA</h2>
  <p class="note" id="guideNote" style="margin-top:0">Une fenêtre « Gaussian-Splatting-Viewer » va s'ouvrir sur l'écran de cette machine
  (si la scène n'est pas encore préparée, l'app le fait d'abord : suivez la console).</p>
  <ol class="guide">
    <li>Dans le panneau <b>Control</b> à droite, cochez <kbd>clickmode</kbd>.</li>
    <li>Faites un <b>clic droit</b> sur l'objet à garder : il s'allume dans l'aperçu.</li>
    <li>Réglez <kbd>ScoreThres</kbd> (0,7 à 0,9) jusqu'à ce que <b>seul l'objet</b> reste coloré.
        <kbd>Scale</kbd> : petit = pièces fines, grand = objet entier.
        Plusieurs objets ? cochez <kbd>multi-clickmode</kbd> et cliquez-les tous.</li>
    <li>Cliquez <kbd>segment3d</kbd> : il ne reste que l'objet en 3D.
        <kbd>roll_back</kbd> annule, <kbd>clear</kbd> recommence.</li>
    <li>Tapez un nom dans la case sous <kbd>save as</kbd>, puis cliquez <kbd>save as</kbd>.</li>
    <li><b>Fermez la fenêtre</b> : le PLY de l'objet apparaît ici, dans « Fichiers PLY produits ».</li>
  </ol>
  <p class="note">Navigation : clic gauche glissé = tourner · molette = zoom · clic milieu glissé = déplacer.</p>
  <div class="row" style="justify-content:flex-end;margin-top:14px">
    <button onclick="hide('guide')">Annuler</button>
    <button class="primary" id="guideGo">C'est parti</button>
  </div>
</div></div>

<div class="modal" id="gallery"><div class="box" style="max-width:1100px">
  <div class="row" style="justify-content:space-between"><h2 style="margin:0" id="galTitle">Masques YOLO</h2><button onclick="hide('gallery')">Fermer</button></div>
  <p class="note">En rouge : ce que COLMAP et l'entraînement ignorent. Si la silhouette dépasse ou est ratée, ajustez puis « Refaire depuis… COLMAP ».</p>
  <div id="galGrid" style="display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:8px"></div>
</div></div>

<div class="modal" id="browse"><div class="box">
  <h2>Choisir sur la machine</h2>
  <div class="row" id="places"></div>
  <div class="note" id="bpath" style="word-break:break-all"></div>
  <ul class="bl" id="blist"></ul>
  <div class="row" style="justify-content:space-between">
    <button onclick="hide('browse')">Fermer</button>
    <button class="primary" onclick="importPath(curPath)">Importer ce dossier</button>
  </div>
</div></div>

<script>
const $ = id => document.getElementById(id);
let S = null, logNext = 0, busy = false, wasRunning = false;
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const size = b => b > 1e9 ? (b/1e9).toFixed(2)+' Go' : b > 1e6 ? (b/1e6).toFixed(1)+' Mo' : Math.round(b/1e3)+' ko';
const show = id => $(id).classList.add('show'), hide = id => $(id).classList.remove('show');
const thenOpt = () => document.querySelector('input[name=then]:checked').value;
// Ne redessine que si le contenu change (sinon un clic peut tomber pendant le rafraichissement)
const setHTML = (id, html) => { const el = $(id); if (el._html !== html) { el.innerHTML = html; el._html = html; } };

async function api(path, body) {
  const r = await fetch(path, body === undefined ? {} : {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
  const j = await r.json().catch(() => ({}));
  if (!r.ok) { alert(j.error || ('Erreur ' + r.status)); throw new Error(j.error); }
  return j;
}

async function refresh() {
  try { S = await api('/api/state'); } catch (e) { return; }
  busy = S.job.running;
  const i = S.installed;
  setHTML('pills',
    `<span class="pill ${S.gpu ? 'ok' : 'no'}">${S.gpu ? '🟢 ' + esc(S.gpu) : '⚠️ Pas de GPU NVIDIA détecté'}</span>` +
    `<span class="pill ${i.tools ? 'ok' : 'no'}">${i.tools ? '✓' : '○'} FFmpeg · COLMAP</span>` +
    `<span class="pill ${i.saga ? 'ok' : 'no'}">${i.saga ? '✓' : '○'} SAGA</span>` +
    `<button onclick="act('install')" ${busy ? 'disabled' : ''} title="Télécharge et installe tout ce qui manque, sans droits admin">${i.tools && i.saga ? 'Vérifier l\'installation' : '⬇ Tout installer'}</button>`);
  renderProjects();
  $('outdir').textContent = S.base + '/output_ply';
  setHTML('outputs', S.outputs.length ? S.outputs.map(f =>
    `<div>📦 <a href="/download/${encodeURIComponent(f.name)}">${esc(f.name)}</a> <span class="kind">${size(f.size)}</span></div>`).join('')
    : '<span class="empty">Rien pour l\'instant.</span>');
}

function renderProjects() {
  if (!S.projects.length) { setHTML('projects', '<p class="empty">Aucun projet : ajoutez une vidéo, des photos ou un PLY ci-dessus.</p>'); return; }
  const dis = busy ? 'disabled' : '';
  setHTML('projects', S.projects.map(p => {
    const n = esc(p.name);
    const st = [
      ['Images', p.input >= 3 || p.images >= 3 || p.kind === 'ply'],
      [p.synthetic ? 'Vues virtuelles' : 'Caméras', p.cameras],
      ['Modèle 3D', !!p.model],
      ['SAGA prêt', !!p.features],
      [`Objets (${p.segs.length})`, p.segs.length > 0],
    ];
    let main;
    if (!p.model) main = `<button class="primary" ${dis} onclick="act('run','${n}')">▶ Créer le PLY</button>`;
    else main = `<button class="primary" ${dis} onclick="gui('${n}')">✂ ${p.features ? 'Ouvrir la découpe SAGA' : 'Découper un objet'}</button>`;
    let more = '';
    if (p.model && !p.features) more += `<button ${dis} onclick="act('saga','${n}')" title="Calcule les masques SAM et les features sans ouvrir la fenêtre">Préparer SAGA seulement</button>`;
    if (!p.model) more += `<button ${dis} onclick="act('saga','${n}')">PLY + préparer SAGA</button>`;
    if (p.final) more += `<button ${dis} onclick="act('clean','${n}')" title="Supprime les gaussiennes parasites (floaters)">🧹 Nettoyer</button>`;
    const redo = [];
    if (p.video) redo.push(['frames', p.kind === 'vidéo 360°' ? 'extraction et découpe 360° (FPS, YOLO…)' : 'extraction des images (FPS…)']);
    if (!p.imported_colmap && !p.legacy && (!p.imported_ply || p.synthetic) && (p.cameras || p.input)) redo.push(['colmap', p.synthetic ? 'vues virtuelles' : 'COLMAP (YOLO, résolution…)']);
    if (!p.imported_ply && p.model) redo.push(['train', 'entraînement 3DGS (itérations, reflets…)']);
    if (p.sam || p.features) redo.push(['saga', 'préparation SAGA']);
    if (redo.length) more += `<select class="redo" ${dis} onchange="reset('${n}', this)"><option value="">↺ Refaire depuis…</option>${redo.map(([v, l]) => `<option value="${v}">${l}</option>`).join('')}</select>`;
    if (p.masks.length) more += `<button class="primary" ${dis} onclick="act('package','${n}')" title="output_ply/${n}/ : scène complète, scène seule (sans les objets) et un .ply par objet. Les masques invalides (vides ou toute la scène) sont écartés.">📦 Dossier final : scène + ${p.masks.length} objet${p.masks.length > 1 ? 's' : ''}</button>`;
    const dl = f => `<div>📦 <a href="/download/${f.split('/').map(encodeURIComponent).join('/')}">${esc(f)}</a></div>`;
    const files = [p.final, ...p.segs].filter(Boolean).map(dl).join('')
      + (p.package.length ? `<div class="note" style="margin:6px 0 2px">Dossier final <b>output_ply/${n}/</b> :</div>` + p.package.map(dl).join('') : '');
    let info = p.kind === 'ply' && !p.cameras ? '<div class="note">PLY seul : pour SAGA, l\'app rend des vues virtuelles de la scène (idéal pour un objet ou une pièce vue de l\'intérieur).</div>' : '';
    if (p.yolo) more += `<button onclick="gallery('${n}')" title="Zones masquées en rouge">👁 Masques YOLO (${p.yolo})</button>`;
    return `<div class="proj"><h3>${n}<span class="kind">${esc(p.kind)}${p.video ? ' · ' + esc(p.video) : ''}</span></h3>
      <div class="steps">${st.map(([l, d]) => `<div class="st ${d ? 'done' : ''}">${d ? '✓ ' : ''}${l}</div>`).join('')}</div>
      <div class="row">${main}${more}</div>${info}${files ? '<div class="files">' + files + '</div>' : ''}</div>`;
  }).join(''));
}

async function act(action, project, extra) {
  await api('/api/action', Object.assign({action, project}, extra || {}));
  logNext = 0; $('con').textContent = ''; poll(); refresh();
}

function gui(p) {
  $('guideNote').style.color = S.display ? '' : 'var(--bad)';
  if (!S.display) $('guideNote').textContent = "Attention : aucun écran détecté pour ce serveur. Lancez l'app depuis la session graphique de la machine pour que la fenêtre SAGA puisse s'ouvrir.";
  $('guideGo').onclick = () => { hide('guide'); act('gui', p); };
  show('guide');
}

async function gallery(p) {
  const j = await api('/api/previews?project=' + encodeURIComponent(p));
  $('galTitle').textContent = `Masques YOLO - ${p} (${j.files.length})`;
  $('galGrid').innerHTML = j.files.map(f =>
    `<a href="/preview/${encodeURIComponent(p)}/${encodeURIComponent(f)}" target="_blank"><img loading="lazy" style="width:100%;border-radius:8px" src="/preview/${encodeURIComponent(p)}/${encodeURIComponent(f)}" title="${esc(f)}"></a>`).join('');
  show('gallery');
}

async function reset(p, sel) {
  const step = sel.value, label = sel.options[sel.selectedIndex].text;
  sel.value = '';
  if (!step || !confirm(`Projet ${p} : effacer les calculs à partir de « ${label} » pour les refaire avec les réglages actuels ?\n(La vidéo, les photos et les PLY importés ne sont jamais effacés.)`)) return;
  await act('reset', p, {step});
}

// ---------- Reglages ----------
let SPEC = [], SAVED = {};
const PRESETS = {
  fast: {GS_FPS: 1, GS_MAX_FRAMES: 120, GS_MAX_IMAGE_SIZE: 1280, GS_ITERS: 7000, GS_SAGA_ITERS: 5000, GS_SYNTH_VIEWS: 150},
  normal: {},
  quality: {GS_FPS: 3, GS_MAX_FRAMES: 400, GS_MAX_IMAGE_SIZE: 'auto', GS_ITERS: 30000, GS_SAGA_ITERS: 10000, GS_SYNTH_VIEWS: 400},
};

async function loadSettings() {
  const j = await api('/api/settings');
  SPEC = j.spec; SAVED = j.values;
  const hw = j.hw || {};
  $('hwNote').textContent = hw.VRAM_MB
    ? `🖥️ Détecté : ${hw.GPU_NAME || 'carte graphique'} (${(hw.VRAM_MB / 1024).toFixed(0)} Go), RAM libre ${(hw.RAM_MB / 1024).toFixed(0)} Go. `
      + `« auto » s'adapte à ce matériel (résolution max ≈ ${hw.AUTO_MAX_IMAGE_SIZE} px) et réduit tout seul les images si la mémoire déborde.`
    : '« auto » s\'adapte à la carte graphique et à la RAM, et réduit tout seul les images si la mémoire déborde.';
  let html = '', group = '';
  for (const s of SPEC) {
    if (s.group !== group) { html += (group ? '</div>' : '') + `<div class="sgroup"><h4>${esc(s.group)}</h4>`; group = s.group; }
    const v = SAVED[s.key], id = 'set_' + s.key;
    let input;
    if (s.type === 'bool') input = `<input type="checkbox" id="${id}" ${String(v) === '1' ? 'checked' : ''}>`;
    else if (s.type === 'choice') input = `<select id="${id}">${s.choices.map(c => `<option ${String(c) === String(v) ? 'selected' : ''}>${c}</option>`).join('')}</select>`;
    else if (s.type === 'text') input = `<input type="text" id="${id}" value="${esc(v)}">`;
    else input = `<input type="number" id="${id}" value="${esc(v)}" min="${s.min}" max="${s.max}" step="${s.type === 'float' ? 'any' : 1}">`;
    html += `<div class="field" id="f_${s.key}"><label for="${id}">${esc(s.label)}</label>${input}<small>${esc(s.help)}</small></div>`;
  }
  $('setForm').innerHTML = html + '</div>';
  $('setForm').querySelectorAll('input,select').forEach(el => el.addEventListener('input', markChanged));
  markChanged();
}

function formValues() {
  const o = {};
  for (const s of SPEC) { const el = $('set_' + s.key); o[s.key] = s.type === 'bool' ? (el.checked ? 1 : 0) : el.value; }
  return o;
}

function markChanged() {
  const v = formValues();
  for (const s of SPEC) $('f_' + s.key).classList.toggle('changed', String(v[s.key]) !== String(SAVED[s.key]));
  $('setSummary').textContent = `${SAVED.GS_FPS} img/s · ${SAVED.GS_ITERS} itér. · max ${SAVED.GS_MAX_FRAMES} images` + (String(SAVED.GS_YOLO) === '1' ? ' · YOLO actif' : '');
}

function preset(name) {
  const vals = Object.assign({}, ...SPEC.map(s => ({[s.key]: s.default})), PRESETS[name]);
  for (const s of SPEC) {
    if (/YOLO|360/.test(s.group) || s.key === 'GS_SH_DEGREE' || s.key === 'GS_SAM_DOWNSAMPLE') continue;  // choix de l'utilisateur gardes
    const el = $('set_' + s.key);
    if (s.type === 'bool') el.checked = String(vals[s.key]) === '1'; else el.value = vals[s.key];
  }
  markChanged();
}

async function saveSettings() {
  const j = await api('/api/settings', formValues());
  SAVED = j.values; markChanged();
  $('setSummary').textContent += '  ✓ enregistré';
}

async function cancelJob() { if (confirm('Annuler la tâche en cours ?')) await api('/api/cancel', {}); }

async function poll() {
  let j;
  try { j = await (await fetch('/api/log?since=' + logNext)).json(); } catch (e) { return; }
  const con = $('con'), stick = con.scrollTop + con.clientHeight >= con.scrollHeight - 30;
  if (j.reset) con.textContent = '';
  if (j.lines.length) con.textContent += j.lines.join('\n') + '\n';
  if (con.textContent.length > 400000) con.textContent = con.textContent.slice(-300000);
  logNext = j.next;
  $('prog').textContent = j.progress || '';
  $('jtitle').textContent = j.title || '';
  const s = $('jstatus');
  if (j.running) { s.textContent = '⏳ En cours…'; s.className = 'status run'; }
  else if (j.rc === 0) { s.textContent = '✅ Terminé'; s.className = 'status ok'; }
  else if (j.rc !== null && j.rc !== undefined) { s.textContent = '❌ Échec : lisez la fin du log'; s.className = 'status bad'; }
  $('cancel').disabled = !j.running;
  if (stick) con.scrollTop = con.scrollHeight;
  if (wasRunning !== j.running) { wasRunning = j.running; refresh(); }
}

// ---------- Envoi de fichiers ----------
function pick(dir) { const el = dir ? $('fDir') : $('fFiles'); el.value = ''; el.click(); }
$('fFiles').onchange = e => send([...e.target.files].map(f => ({file: f, rel: f.name})));
$('fDir').onchange = e => send([...e.target.files].map(f => ({file: f, rel: f.webkitRelativePath || f.name})));

const drop = $('drop');
['dragenter', 'dragover'].forEach(t => drop.addEventListener(t, e => { e.preventDefault(); drop.classList.add('over'); }));
['dragleave', 'drop'].forEach(t => drop.addEventListener(t, e => { e.preventDefault(); drop.classList.remove('over'); }));
drop.addEventListener('drop', async e => {
  const items = [...e.dataTransfer.items].map(i => i.webkitGetAsEntry && i.webkitGetAsEntry()).filter(Boolean);
  const out = [];
  async function walk(entry, prefix) {
    if (entry.isFile) await new Promise(r => entry.file(f => { out.push({file: f, rel: prefix + f.name}); r(); }, r));
    else if (entry.isDirectory) {
      const reader = entry.createReader();
      let batch;
      do {
        batch = await new Promise(r => reader.readEntries(r, () => r([])));
        for (const c of batch) await walk(c, prefix + entry.name + '/');
      } while (batch.length);
    }
  }
  if (items.length) { for (const it of items) await walk(it, ''); }
  else out.push(...[...e.dataTransfer.files].map(f => ({file: f, rel: f.name})));
  send(out);
});

function autoName(list) {
  const first = list[0].rel;
  if (first.includes('/') && list.every(x => x.rel.split('/')[0] === first.split('/')[0])) return first.split('/')[0];
  return list.length === 1 || /\.(mp4|mov|mkv|avi|webm|m4v|insv|ply|zip)$/i.test(first) ? first.replace(/^.*\//, '').replace(/\.[^.]+$/, '') : 'photos';
}

async function send(list) {
  if (!list.length) return;
  if (busy) { alert('Une tâche est en cours : attendez la fin ou annulez-la.'); return; }
  const token = (crypto.randomUUID ? crypto.randomUUID() : Date.now() + '' + Math.random()).replace(/[^A-Za-z0-9]/g, '');
  const name = $('pname').value.trim() || autoName(list);
  const total = list.reduce((a, x) => a + x.file.size, 0);
  let done = 0;
  $('bar').style.display = 'block';
  for (let k = 0; k < list.length; k++) {
    const x = list[k];
    $('upnote').textContent = `Envoi ${k + 1}/${list.length} : ${x.rel} (${size(total)} au total)`;
    await new Promise((res, rej) => {
      const r = new XMLHttpRequest();
      r.open('POST', `/api/upload?token=${token}&rel=${encodeURIComponent(x.rel)}`);
      r.upload.onprogress = ev => { $('bar').firstElementChild.style.width = ((done + ev.loaded) / total * 100) + '%'; };
      r.onload = () => r.status === 200 ? res() : rej(new Error(r.responseText));
      r.onerror = () => rej(new Error('connexion perdue'));
      r.send(x.file);
    }).catch(e => { alert('Envoi échoué : ' + e.message); throw e; });
    done += x.file.size;
  }
  $('upnote').textContent = 'Envoi terminé, import en cours (voir la console)…';
  await api('/api/import', {token, name, then: thenOpt()});
  $('pname').value = '';
  setTimeout(() => { $('bar').style.display = 'none'; $('upnote').textContent = ''; }, 1500);
  logNext = 0; $('con').textContent = ''; poll(); refresh();
}

// ---------- Choisir un fichier deja present sur la machine ----------
let curPath = '';
async function openBrowse(path) {
  const j = await api('/api/browse?path=' + encodeURIComponent(path || curPath || ''));
  curPath = j.path;
  $('bpath').textContent = j.path;
  $('places').innerHTML = j.places.map(p => `<button onclick="openBrowse('${esc(p).replace(/'/g, "\\'")}')">${esc(p)}</button>`).join('');
  const usable = e => e.dir || /\.(mp4|mov|mkv|avi|webm|m4v|insv|ply|zip|tar|tgz|jpe?g|png)$/i.test(e.name);
  $('blist').innerHTML = `<li data-up="1">⬆ ..</li>` + j.entries.filter(usable).map(e =>
    `<li data-n="${esc(e.name)}" data-d="${e.dir ? 1 : 0}"><span>${e.dir ? '📁' : '📄'} ${esc(e.name)}</span><small>${e.dir ? '' : size(e.size)}</small></li>`).join('');
  $('blist').querySelectorAll('li').forEach(li => li.onclick = () => {
    if (li.dataset.up) return openBrowse(j.parent);
    const full = j.path.replace(/\/$/, '') + '/' + li.dataset.n;
    if (li.dataset.d === '1') openBrowse(full); else importPath(full);
  });
  show('browse');
}

async function importPath(p) {
  if (!confirm('Importer ' + p + ' ?')) return;
  hide('browse');
  await api('/api/import', {path: p, name: $('pname').value.trim(), then: thenOpt()});
  $('pname').value = ''; logNext = 0; $('con').textContent = ''; poll(); refresh();
}

// ---------- Theme ----------
const THEMES = {auto: '🌓 Auto', dark: '🌙 Sombre', light: '☀️ Clair'};
function applyTheme(t) {
  if (t === 'auto') delete document.documentElement.dataset.theme; else document.documentElement.dataset.theme = t;
  $('themeBtn').textContent = THEMES[t];
  try { localStorage.setItem('ga_theme', t); } catch (e) {}
}
function cycleTheme() {
  const cur = document.documentElement.dataset.theme || 'auto';
  applyTheme({auto: 'dark', dark: 'light', light: 'auto'}[cur]);
}
applyTheme(document.documentElement.dataset.theme || 'auto');

// ---------- Console plein ecran ----------
function toggleFull(force) {
  const el = $('console'), on = force !== undefined ? force : !el.classList.contains('full');
  el.classList.toggle('full', on);
  $('fullBtn').textContent = on ? '✕ Quitter le plein écran' : '⛶ Plein écran';
  document.body.style.overflow = on ? 'hidden' : '';
  // vrai plein ecran du navigateur si possible (sinon la console couvre juste la fenetre)
  try {
    if (on && !document.fullscreenElement && el.requestFullscreen) el.requestFullscreen().catch(() => {});
    if (!on && document.fullscreenElement) document.exitFullscreen().catch(() => {});
  } catch (e) {}
  const con = $('con'); con.scrollTop = con.scrollHeight;
}
document.addEventListener('fullscreenchange', () => { if (!document.fullscreenElement && $('console').classList.contains('full')) toggleFull(false); });
document.addEventListener('keydown', e => { if (e.key === 'Escape' && $('console').classList.contains('full')) toggleFull(false); });
let conSize = 14;
try { conSize = +localStorage.getItem('ga_confont') || 14; } catch (e) {}
function conFont(d) {
  conSize = Math.min(24, Math.max(10, conSize + d));
  $('console').style.setProperty('--confont', conSize + 'px');
  try { localStorage.setItem('ga_confont', conSize); } catch (e) {}
}
conFont(0);

refresh(); poll(); loadSettings();
setInterval(refresh, 4000);
setInterval(poll, 1000);
</script>
</body>
</html>
"""


def main():
    global APP, BASE, WORK, OUT, STATE, GPU
    ap = argparse.ArgumentParser(description="Interface web de Gaussian App")
    ap.add_argument("--app", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 pour y acceder depuis un autre PC (reseau de confiance uniquement)")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    APP, BASE = os.path.abspath(a.app), os.path.abspath(a.base)
    WORK, OUT, STATE = os.path.join(BASE, "workspace"), os.path.join(BASE, "output_ply"), os.path.join(BASE, ".gaussian_app")
    for d in (WORK, OUT, STATE):
        os.makedirs(d, exist_ok=True)
    try:
        GPU = subprocess.run(["nvidia-smi", "--query-gpu=gpu_name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
    except (OSError, IndexError, subprocess.SubprocessError):
        GPU = ""

    port = a.port
    for port in range(a.port, a.port + 20):   # premier port libre
        try:
            srv = ThreadingHTTPServer((a.host, port), Handler)
            break
        except OSError:
            continue
    else:
        raise SystemExit("Aucun port libre.")
    url = f"http://{'localhost' if a.host in ('127.0.0.1', '0.0.0.0') else a.host}:{port}"
    print("=" * 70)
    print(f"  Gaussian App est ouverte dans votre navigateur : {url}")
    print("  (laissez ce terminal ouvert ; Ctrl+C pour quitter)")
    print("=" * 70, flush=True)
    if not a.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        JOB.cancel()
        print("\nAu revoir.")


if __name__ == "__main__":
    main()
