"""Registre des 3 modèles Aura servis par l'API : CamemBERT (modèle retenu), SVM et CNN.

Les modèles sont lus dans ml/models/ :
  - CamemBERT : ensemble choisi en phase 2 (ml/results/p2_choix.json + ml/models/p2_*/final),
                téléchargé automatiquement depuis le Hugging Face Hub s'il manque (ml/scripts/hf_modeles.py)
  - SVM       : ml/models/svm.joblib   (créé par ml/notebook_aura.ipynb)
  - CNN       : ml/models/cnn/         (créé par ml/notebook_aura.ipynb)
Chargement à la demande puis mise en cache. CPU par défaut ; AURA_DEVICE=cuda pour le GPU.

Chaque modèle renvoie : catégorie, probabilité, probabilités des 4 classes et un score d'aura (-100..100).
Pour CamemBERT (score continu), les « probabilités » sont une distribution douce construite à partir des seuils
(distance du score à l'intervalle de chaque classe) : la classe affichée a toujours la barre la plus haute ;
pour le SVM et le CNN, le score est l'espérance Σ p_k · centre_k.
"""
import hashlib
import json
import os
import sys
import threading
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ML = ROOT / "ml"
MODELS = ML / "models"
sys.path.insert(0, str(ML / "scripts"))

ORDRE = ["Anti-Aura", "Aura Correcte", "Forte Aura", "Aura Divine"]
CENTRES = np.array([-55.0, 7.0, 50.0, 87.0])
DEVICE = os.environ.get("AURA_DEVICE", "cpu")
# AURA_DTYPE=bfloat16 divise par 2 la mémoire de CamemBERT (~2 Go au lieu de 3,3) sans changer ses prédictions
# sur test_reel (kappa identique) ; utilisé pour Streamlit Community Cloud, limité en RAM.
DTYPE = os.environ.get("AURA_DTYPE", "float32")


def proba_depuis_score(s, temperature=18.0):
    """Distribution douce autour des CENTRES des classes (modèle démo, où la classe est l'argmax)."""
    z = -np.abs(s - CENTRES) / temperature
    p = np.exp(z - z.max())
    return p / p.sum()


def proba_depuis_seuils(s, seuils, temperature=18.0):
    """Distribution douce cohérente avec les SEUILS qui décident la classe (np.digitize) : chaque classe est
    pénalisée par la distance du score à SON intervalle [seuil bas, seuil haut[. La classe qui contient le score
    est à distance 0, elle a donc toujours la plus forte probabilité (contrairement à la distance aux centres :
    avec les seuils [-12, 36, 69], un score de -22 est Anti-Aura mais plus proche du centre de « Correcte »)."""
    bornes = np.concatenate([[-np.inf], np.asarray(seuils, dtype=float), [np.inf]])
    bas, haut = bornes[:-1], bornes[1:]
    # intervalle semi-ouvert comme np.digitize : un score pile sur un seuil appartient à la classe du dessus
    dist = np.where(s < bas, bas - s, np.where(s >= haut, s - haut + 1e-6, 0.0))
    z = -dist / temperature
    p = np.exp(z - z.max())
    return p / p.sum()


# ------------------------------------------------------------------ modèles
class CamembertEnsemble:
    """Moyenne pondérée des scores de plusieurs CamemBERT + seuils choisis sur la validation."""
    def __init__(self, retenu):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch, self.seuils, self.parts = torch, tuple(retenu["seuils"]), []
        for nom, w in retenu["composants"].items():
            if nom == "tfidf" or w <= 0: continue
            d = MODELS / nom / "final"
            m = AutoModelForSequenceClassification.from_pretrained(d, dtype=getattr(torch, DTYPE))
            self.parts.append((w, AutoTokenizer.from_pretrained(d), m.to(DEVICE).eval()))
        tot = sum(w for w, _, _ in self.parts); self.parts = [(w / tot, t, m) for w, t, m in self.parts]

    def predire(self, text):
        s = 0.0
        with self.torch.no_grad():
            for w, tok, m in self.parts:
                o = m(**tok([text], truncation=True, max_length=256, return_tensors="pt").to(DEVICE)).logits[0].float().cpu().numpy()
                if o.size == 1: s += w * float(o[0]) * 100
                else: e = np.exp(o - o.max()); s += w * float((e / e.sum()) @ CENTRES)
        return int(np.digitize(s, self.seuils)), proba_depuis_seuils(s, self.seuils), s


class SvmModel:
    def __init__(self):
        import joblib
        self.pipe = joblib.load(MODELS / "svm.joblib")

    def predire(self, text):
        p = self.pipe.predict_proba([text])[0]; return int(p.argmax()), p, float(p @ CENTRES)


class CnnModel:
    def __init__(self):
        import cnn
        self.f = cnn.charger(str(MODELS / "cnn"), DEVICE)

    def predire(self, text):
        p = self.f([text])[0]; return int(p.argmax()), p, float(p @ CENTRES)


class DemoModel:
    def predire(self, text):
        s = hashlib.sha256(text.encode()).digest()[0] / 255 * 200 - 100
        p = proba_depuis_score(s); return int(p.argmax()), p, s


# ------------------------------------------------------------------ registre
def _kappas():
    """Kappa test de chaque modèle, écrit par le notebook (results/comparaison_3_modeles.json)."""
    f = ML / "results" / "comparaison_3_modeles.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def _depuis_hugging_face():
    """CamemBERT (~2,6 Go) n'est pas sur GitHub : s'il manque en local, on le récupère sur le Hugging Face Hub
    (dépôt dans AURA_HF_REPO ou ml/hf_repo.txt ; AURA_HF_REPO= vide pour désactiver)."""
    try:
        import hf_modeles
        if hf_modeles.manquants() and hf_modeles.repo_id():
            hf_modeles.telecharger()
    except Exception as e:  # hors ligne, dépôt introuvable… : on continue avec SVM / CNN
        print(f"[aura] CamemBERT indisponible ({e.__class__.__name__}: {e})", flush=True)


def decouvrir():
    k = _kappas()
    desc = lambda base, cle: base + (f" — kappa test {k[cle]:.3f}".replace(".", ",") if cle in k else "")
    reg = []
    choix_f = ML / "results" / "p2_choix.json"
    if choix_f.exists():
        choix = json.loads(choix_f.read_text(encoding="utf-8")); retenu = choix["candidats"][choix["meilleur"]]
        _depuis_hugging_face()
        if all((MODELS / n / "final").exists() for n, w in retenu["composants"].items() if n != "tfidf" and w > 0):
            reg.append(dict(id="camembert", label="⭐ CamemBERT (modèle retenu)", famille="pré-entraîné",
                            description=desc("Pré-entraîné sur le français, ensemble large + base", "camembert"),
                            loader=lambda r=retenu: CamembertEnsemble(r)))
    if (MODELS / "svm.joblib").exists():
        reg.append(dict(id="svm", label="SVM", famille="machine learning",
                        description=desc("TF-IDF + SVM linéaire, ultra rapide", "svm"), loader=SvmModel))
    if (MODELS / "cnn" / "modele.pt").exists():
        reg.append(dict(id="cnn", label="CNN", famille="réseau de neurones",
                        description=desc("Réseau convolutif appris de zéro", "cnn"), loader=CnnModel))
    if not reg:
        reg.append(dict(id="demo", label="Démo (aucun modèle entraîné)", famille="démo", description="Prédictions factices", loader=DemoModel))
    return {e["id"]: e for e in reg}


class Registry:
    def __init__(self):
        self.entries = decouvrir()
        self.cache, self.lock = {}, threading.Lock()

    @property
    def default_id(self): return next(iter(self.entries))
    @property
    def is_demo(self): return "demo" in self.entries

    def liste(self):
        return [{k: e[k] for k in ("id", "label", "description", "famille")} | {"default": e["id"] == self.default_id}
                for e in self.entries.values()]

    def get(self, model_id):
        model_id = model_id if model_id in self.entries else self.default_id
        with self.lock:
            if model_id not in self.cache:
                self.cache[model_id] = self.entries[model_id]["loader"]()
        return model_id, self.cache[model_id]

    def predict(self, model_id, text):
        model_id, m = self.get(model_id)
        k, proba, score = m.predire(text)
        ranked = sorted(zip(ORDRE, proba), key=lambda x: x[1], reverse=True)
        return {
            "category": ORDRE[k], "probability": float(proba[k]), "score": round(max(-100, min(100, score))),
            "probabilities": [{"label": l, "probability": float(p)} for l, p in ranked],
            "model": {"id": model_id, "label": self.entries[model_id]["label"]}, "demo": model_id == "demo",
        }
