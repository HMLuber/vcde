# Echtzeit-Objekterkennung in Videostreams

> Vertiefungsthema 1 · Visual Computing · Hochschule München · **Quarto-Site**

Eine browser-basierte Demo zur **Echtzeit-Objekterkennung (Localization + Classification) in Videostreams** mit ≥ 24–30 FPS. Die komplette Inferenz läuft clientseitig im Browser — kein Server, keine API, keine Datei verlässt das Endgerät.

## ✨ Features

- 📹 **Video-Upload** per Drag & Drop oder Datei-Auswahl (mp4, webm, mov, …)
- 📷 **Webcam-Modus** für Live-Inferenz
- 🎯 **Bounding Boxes + Klassen + Konfidenz** pixelgenau über das Video gezeichnet
- 📊 **Live-Telemetrie:** FPS (gleitender Mittelwert), Inferenzzeit pro Frame, Frame-Counter, Auflösung, Backend
- 🎚️ Konfidenz-Schwellenwert live einstellbar
- 📚 Anzeige aller erkannten Klassen mit Häufigkeit und Konfidenz

## 📁 Projektstruktur

```
.
├── _quarto.yml       # Quarto-Projektkonfiguration (Theme: none, page-layout: custom)
├── index.qmd         # Hauptseite: Front-Matter + raw HTML-Block
├── style.css         # Custom Stylesheet (dark editorial / lab look)
├── app.js            # Detection-Loop, Rendering, FPS-Logik
├── .gitignore        # Build-Artefakte ausgeschlossen
└── README.md         # Diese Datei
```

Beim Rendern erzeugt Quarto den Output-Ordner `_site/`. Dieser ist
gitignored — wir publishen ihn separat (siehe unten).

## 🛠️ Verwendete Technologien

| Komponente   | Technologie                          | Funktion                                            |
| ------------ | ------------------------------------ | --------------------------------------------------- |
| Build-System | **Quarto**                           | Static-Site-Generator, rendert `.qmd` → HTML        |
| Modell       | **COCO-SSD** (`lite_mobilenet_v2`)   | Object Detector, vortrainiert auf COCO (80 Klassen) |
| Runtime      | **TensorFlow.js** + **WebGL Backend**| GPU-beschleunigte Inferenz im Browser               |
| Rendering    | HTML5 `<canvas>` Overlay             | Bounding Boxes synchron zum Video                   |
| Hosting      | GitHub Pages                         | Statisches Hosting, kein Server                     |

> **Hinweis zu YOLO:** Die Aufgabenstellung listet *OpenCV* und *YOLO* als
> Vorschläge. Im Browser ist YOLO direkt nicht lauffähig (PyTorch-/Darknet-
> Architektur). COCO-SSD löst exakt dasselbe Problem (Single-Shot Detector
> → Localization + Classification in einem Forward-Pass) und erreicht im
> Browser stabil ≥ 30 FPS. Möchte man explizit YOLO einsetzen, kann das
> Modell mit [`onnxruntime-web`](https://onnxruntime.ai/docs/tutorials/web/)
> und einer ONNX-Export-Datei (`yolov8n.onnx`) angebunden werden — siehe
> Abschnitt *"Auf YOLO umstellen"* unten.

## 🚀 Setup

### 1. Quarto installieren

Falls noch nicht vorhanden — Download von <https://quarto.org/docs/get-started/>.
Verifizieren mit:

```bash
quarto --version
```

### 2. Lokal entwickeln

```bash
quarto preview
```

Öffnet automatisch einen lokalen Dev-Server (i. d. R. `http://localhost:4848`)
mit Hot-Reload. Änderungen an `index.qmd`, `style.css` oder `app.js` werden
sofort übernommen.

### 3. Render-Build

```bash
quarto render
```

Erzeugt den Output in `_site/`. Diesen Ordner kann man als Static Site
ausliefern.

## 🌐 Deployment auf GitHub Pages

**Empfohlen: `quarto publish gh-pages`** — Quarto pflegt einen separaten
`gh-pages`-Branch und pusht den gerenderten Output automatisch dorthin.

### Schritt für Schritt

```bash
# 1. Repo initialisieren (falls noch nicht geschehen)
git init
git add .
git commit -m "Initial Quarto project"
git branch -M main
git remote add origin git@github.com:<USERNAME>/<REPO>.git
git push -u origin main

# 2. Auf GitHub: Settings → Pages → Source = "gh-pages" / root
#    (Branch existiert noch nicht — wird im nächsten Schritt angelegt.)

# 3. Quarto rendern und publishen
quarto publish gh-pages
```

Beim ersten Aufruf fragt Quarto, ob ein `gh-pages`-Branch erstellt werden
soll → bestätigen mit *yes*. Quarto rendert, committed in `gh-pages`, pusht.
Nach 1–2 Minuten ist die Seite live unter:

```
https://<USERNAME>.github.io/<REPO>/
```

### Alternative: Action-basiert (CI)

Wer ohne lokales Rendern publishen will, kann eine GitHub-Action einrichten
(siehe <https://quarto.org/docs/publishing/github-pages.html#publish-action>).
Für ein einzelnes Universitätsprojekt ist `quarto publish gh-pages` aber
deutlich einfacher.

## 📈 Wie wird die FPS-Vorgabe erfüllt?

Die App misst die **echte End-to-End FPS** (RAF-Frequenz inkl. Inferenz +
Rendering) als gleitenden Mittelwert über 30 Frames. Auf moderner Hardware
liefert das Setup typischerweise:

| Hardware                                    | Inferenz pro Frame | FPS      |
| ------------------------------------------- | ------------------ | -------- |
| Desktop (RTX 3060 / Apple M1)               | ~ 8–15 ms          | 50–60+   |
| Laptop (Intel UHD)                          | ~ 25–35 ms         | 28–40    |
| Mid-Range Smartphone                        | ~ 30–45 ms         | 22–32    |

Falls die FPS unter 24 fällt, erscheint der Wert **rot**; im Zielband
24–30 grün-gelb; darüber grün. Der Schwellenwert (Mindest-Konfidenz) kann
in der Sidebar verändert werden.

## 🗂️ Erkannte Klassen

Das Modell erkennt **80 COCO-Klassen**, u. a.:

- Verkehr: `person`, `bicycle`, `car`, `motorcycle`, `bus`, `truck`,
  `traffic light`, `stop sign`, `fire hydrant`, `parking meter`
- Tiere: `dog`, `cat`, `bird`, `horse`, `sheep`, `cow`
- Alltag: `backpack`, `umbrella`, `bottle`, `cup`, `chair`, `laptop`,
  `cell phone`, `book`, …

Damit ist die Demo besonders gut geeignet für **Verkehrs-Videos** —
exakt das Beispiel aus der Aufgabenstellung.

## 🧠 Funktionsweise (Kurzfassung)

```
┌─────────────┐    ┌──────────┐    ┌────────────┐    ┌─────────────┐
│  <video>    │ -> │  COCO-   │ -> │  Filter    │ -> │  Canvas-    │
│  Frame      │    │  SSD     │    │  Threshold │    │  Overlay    │
└─────────────┘    └──────────┘    └────────────┘    └─────────────┘
       │                                                      ^
       └──────── requestAnimationFrame-Loop ──────────────────┘
```

Pro Animation-Frame: aktuelles Video-Frame an das Modell übergeben →
Predictions (`{bbox, class, score}`) zurück → Filtern nach Konfidenz →
Auf `<canvas>` zeichnen, das pixelgenau über dem `<video>` liegt.

## 🔄 Auf YOLO umstellen (optional)

1. YOLOv8-Modell nach ONNX exportieren:
   ```bash
   pip install ultralytics
   yolo export model=yolov8n.pt format=onnx opset=12
   # → yolov8n.onnx (~ 12 MB)
   ```
2. Datei `yolov8n.onnx` ins Repo legen (in `_quarto.yml` als Resource
   ergänzen, damit sie mitkopiert wird).
3. In `index.qmd` die TF.js-Skripte austauschen gegen:
   ```html
   <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>
   ```
4. In `app.js` `model.detect(video)` durch eine ONNX-Inference-Session
   ersetzen und Non-Maximum-Suppression + Sigmoid-Postprocessing
   ergänzen.

Die restliche Architektur (Detection-Loop, FPS-Messung, Rendering) bleibt
identisch.

## 📝 Lizenz / Quellen

- COCO-SSD-Modell: Apache 2.0 (Google / TensorFlow.js Models)
- TensorFlow.js: Apache 2.0
- Beispielbild aus der Aufgabe: `huggingface.co/spaces/atalaydenknalbant/Yolov13`

---

**Visual Computing (DE) · Hochschule München**
