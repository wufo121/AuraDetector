# Aura Detector

Projet final du Mastère Dev, Data & IA : un modèle qui lit un texte en français et dit quelle « aura » il dégage.

On voulait un projet de NLP un peu fun, alors on est partis d'une question bête : est-ce qu'une IA peut sentir si une phrase a de l'aura ? Une réplique qui claque, ou au contraire une phrase gênante qui sent le try-hard. C'était beaucoup plus dur que prévu, mais on a appris plein de choses.

Le dépôt contient un notebook qui compare trois modèles (SVM, CNN, CamemBERT), une WebApp (FastAPI + page web, et une version Streamlit) où on colle un texte ou dépose un fichier, et un « booster d'aura » qui réécrit un texte pour lui en donner plus.

## 1. C'est quoi l'aura ?

| Classe | Score | Exemple |
|---|---|---|
| Anti-Aura | −100 à −11 | « dab dab, six-seven » |
| Aura Correcte | −10 à +24 | « Le train a eu une heure de retard ce matin. » |
| Forte Aura | +25 à +74 | « Vous avez la mémoire courte et le discours long ! » |
| Aura Divine | +75 à +100 | « L'aura, c'est comme le wifi du voisin : ceux qui en parlent le plus sont ceux qui ne l'ont pas. » |

Notre règle : on juge la façon de dire les choses, jamais l'opinion, et on note l'aura de celui qui écrit. Une pique bien tournée a de l'aura, une insulte sèche non.

Chaque texte a reçu cinq notes entre −100 et +100 : Rizz (R), Statut (S), Originalité (O), Beauté du style (B) et Poésie (P).

```
Score = (6·R + 4·S + 3·O + 3·B + 2·P) / 18
```

Les modèles ne voient jamais ces notes, juste le texte.

## 2. Lancer le projet de A à Z

Il faut Python 3.10 ou plus et Git. Un GPU est utile seulement pour réentraîner CamemBERT.

```bash
# 1. Récupérer le projet
git clone https://github.com/wufo121/AuraDetector.git
cd AuraDetector

# 2. Créer et activer l'environnement
python3 -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate

# 3. Installer les dépendances
pip install --upgrade pip
pip install -r requirements.txt

# 4. Lancer la WebApp, puis ouvrir http://localhost:8000
uvicorn backend.app:app --reload

# 5. Ou lancer la version Streamlit (http://localhost:8501)
streamlit run streamlit_app.py

# 6. Ouvrir le notebook
pip install notebook
cd ml
jupyter notebook ../notebook_aura_final.ipynb
```

Au premier lancement, l'API télécharge les modèles CamemBERT depuis Hugging Face (environ 2,6 Go). Sans internet, l'appli marche quand même avec le SVM et le CNN. Le notebook ne réentraîne pas CamemBERT (trop long), il recharge les résultats.

Quelques options utiles :
- `AURA_DEVICE=cuda` pour utiliser le GPU, `AURA_DTYPE=bfloat16` pour diviser la mémoire par deux
- `HF_TOKEN` si le dépôt des modèles est privé
- si `ModuleNotFoundError` : l'environnement n'est pas activé (étape 2)
- si le port 8000 est pris : `uvicorn backend.app:app --reload --port 8001`

## 3. Organisation du dépôt

```
AuraDetector/
├── notebook_aura_final.ipynb   ← le notebook rendu
├── streamlit_app.py            ← version Streamlit
├── backend/                    ← l'API (app.py, extract.py, model.py, ameliorer.py)
├── frontend/                   ← la page web
├── Assets/                     ← les GIFs
└── ml/                         ← données, scripts, modèles, résultats
```

## 4. Les données

Il existe aucun dataset « aura », donc on a du le construire nous-mêmes :

- un **train** d'un peu moins de 10 000 textes, surtout générés par script (poésie, argot internet, politique, SMS, bios…), plus des textes écrits par Grok, des battles de rap inventées et de vraies répliques du domaine public (Cyrano, Molière…) ;
- une **validation** de quelques centaines de textes écrits dans le style des vrais textes ;
- un **test** d'environ 300 vrais textes (Assemblée nationale, théâtre, poésie, conférences de presse, tweets de marques), regardé une seule fois à la fin.

L'idée dont on est le plus fiers, ce sont les paires contrastées : deux textes sur le même sujet, avec presque les mêmes mots mais une aura opposée. Au début le modèle apprenait le thème (« politique = Anti-Aura ») au lieu de la forme, et ces paires l'obligent à regarder comment c'est dit.

On a aussi vérifié qu'aucun texte du train ne recopiait le test (data leakage).

## 5. Les modèles et les résultats

Comme plus de la moitié du test est « Aura Correcte », l'accuracy veut pas dire grand-chose. On utilise le **kappa quadratique** (1 = parfait, 0 = hasard), qui punit plus fort les grosses erreurs, genre confondre Anti et Divine.

| Modèle | Kappa test | F1-macro |
|---|---|---|
| Toujours « Correcte » | 0,00 | 0,18 |
| SVM + TF-IDF | 0,15 | 0,30 |
| CNN (Kim, 2014) | ~0,23 | 0,31 |
| Ensemble de 4 CamemBERT | 0,39 | 0,45 |

- **SVM :** TF-IDF sur les mots et sur des morceaux de lettres, hyperparamètres choisis avec `GridSearchCV`.
- **CNN :** embeddings appris de zéro, des filtres sur 1 à 3 mots, du dropout et un early stopping.
- **CamemBERT :** un modèle déjà pré-entraîné sur du français, qu'on fine-tune sur nos textes. Après 11 entraînements, on a gardé une moyenne de 4 modèles (1 large en régression et 3 base en classification), choisie uniquement sur la validation.

CamemBERT gagne nettement, et par rapport à la baseline du début (0,05), on a multiplié le kappa par presque 8. Mais tous les modèles chutent entre la validation (0,95) et le test (0,39), parce que nos textes générés ressemblent pas assez aux vrais. Le théâtre et la poésie restent durs : « Waterloo ! Waterloo ! Waterloo ! morne plaine ! » est pris pour du try-hard à cause de la répétition.

## 6. La WebApp

Le front envoie le texte ou le fichier à `POST /api/predict`. L'API extrait le texte, choisit le modèle et renvoie la classe, le score et les probabilités. La page affiche le résultat avec un GIF.

Le booster d'aura (`/api/improve`) teste plein de petites variantes du texte (enlever, remplacer ou ajouter un mot), les fait noter par notre modèle et garde la meilleure, sans changer le sens.

### Zoom : comment on extrait le texte d'un document

Notre prof nous avait demandé comment on récupère le texte d'un PDF ou d'un Word, et on avait pas su répondre. En creusant, on a vu que c'est plus interressant que prévu : chaque format range son texte différemment.

- **PDF :** il n'y a pas de phrases, juste des instructions pour l'imprimante du genre « écris ces caractères à telle position ». pypdf lit ces instructions et remet les morceaux dans l'ordre.
- **Word et PowerPoint :** ce sont en fait des archives ZIP remplies de fichiers XML (on peut les renommer en `.zip` pour voir). python-docx et python-pptx lisent ce XML.
- **HTML :** on garde le texte entre les balises et on ignore le code (`<script>`, `<style>`).
- **JSON :** on parcourt la structure et on garde seulement les chaînes.
- **TXT :** c'est déjà du texte, mais il faut deviner l'encodage pour pas casser les accents.

```python
import io
import json
from html.parser import HTMLParser


def extraire_texte(contenu: bytes, nom_fichier: str) -> str:
    """Renvoie le texte brut d'un document, quel que soit son format."""
    extension = nom_fichier.lower().rsplit(".", 1)[-1]

    if extension == "pdf" or contenu.startswith(b"%PDF"):
        from pypdf import PdfReader
        pages = PdfReader(io.BytesIO(contenu)).pages
        return "\n".join(page.extract_text() or "" for page in pages)

    if extension == "docx":
        from docx import Document
        doc = Document(io.BytesIO(contenu))
        lignes = [p.text for p in doc.paragraphs]
        for tableau in doc.tables:
            for ligne in tableau.rows:
                lignes += [cellule.text for cellule in ligne.cells]
        return "\n".join(l for l in lignes if l.strip())

    if extension == "pptx":
        from pptx import Presentation
        lignes = []
        for diapo in Presentation(io.BytesIO(contenu)).slides:
            for forme in diapo.shapes:
                if forme.has_text_frame:
                    lignes += [p.text for p in forme.text_frame.paragraphs]
        return "\n".join(l for l in lignes if l.strip())

    texte = decoder(contenu)
    if extension in ("html", "htm"):
        lecteur = LecteurHTML()
        lecteur.feed(texte)
        return "\n".join(lecteur.morceaux)
    if extension == "json":
        return "\n".join(chaines_json(json.loads(texte)))
    return texte


def decoder(contenu: bytes) -> str:
    """UTF-8, puis encodage Windows, puis latin-1 (qui ne plante jamais)."""
    for encodage in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return contenu.decode(encodage)
        except UnicodeDecodeError:
            continue


class LecteurHTML(HTMLParser):
    """Garde le texte entre les balises, sauf dans <script> et <style>."""
    def __init__(self):
        super().__init__()
        self.morceaux, self.ignorer = [], 0

    def handle_starttag(self, balise, attributs):
        if balise in ("script", "style"):
            self.ignorer += 1

    def handle_endtag(self, balise):
        if balise in ("script", "style") and self.ignorer:
            self.ignorer -= 1

    def handle_data(self, donnees):
        if not self.ignorer and donnees.strip():
            self.morceaux.append(donnees.strip())


def chaines_json(objet):
    """Parcourt dictionnaires et listes et renvoie toutes les chaînes."""
    if isinstance(objet, str):
        yield objet
    elif isinstance(objet, dict):
        for valeur in objet.values():
            yield from chaines_json(valeur)
    elif isinstance(objet, list):
        for element in objet:
            yield from chaines_json(element)
```

En gros : on reconnaît le format grâce à l'extension (et à la signature `%PDF` pour les PDF mal nommés), on lit le fichier en mémoire avec `io.BytesIO` pour éviter un fichier temporaire, puis on applique la bonne méthode. Pour le JSON, `chaines_json` est récursive : elle s'appelle elle-même sur chaque élément d'une liste ou d'un dictionnaire. On l'a testée sur un fichier de chaque format et le texte ressort bien.

Les limites : un PDF scanné ne contient que des images, il faudrait de l'OCR (Tesseract). Un PDF en colonnes peut mélanger les lignes. Et les vieux `.doc` et `.ppt` ne sont pas des ZIP, donc ces librairies savent pas les lire.

## 7. Ce qu'on retient

- Le choix du modèle compte plus que les réglages : passer à CamemBERT a fait beaucoup plus que tous les hyperparamètres.
- Les données comptent autant que le modèle, et notre validation était trop facile.
- Nos étiquettes viennent d'IA, pas d'humains, et deux IA ne sont d'accord qu'à 0,61 de kappa. L'aura c'est subjectif.

## 8. Utilisation de l'IA

On s'est beaucoup aidés de Claude (Claude Code), et on préfère le dire clairement. Les commits concernés le mentionnent (`Co-Authored-By`). Claude a écrit une grande partie des scripts, généré les textes synthétiques, les paires contrastées et les battles, et aidé sur la WebApp.

De notre côté, on a apporté l'idée, la définition de l'aura, la formule, les classes et les phrases de test, et on a fait les choix entre les modèles, la relecture et l'analyse des résultats. On a réécrit le notebook final avec nos propres explications pour bien comprendre chaque étape.
