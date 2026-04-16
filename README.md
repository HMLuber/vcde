# GitHub Pages Object Detection Demo

Diese Website ist jetzt als statische GitHub Pages-Anwendung angelegt.
Sie verwendet OpenCV.js im Browser und führt ein YOLOv5n-ONNX-Modell lokal im Nutzergerät aus.

## Was diese Seite macht
- Lädt das leichte YOLOv5n-Modell aus `models/yolov5n.onnx`
- Verwendet OpenCV.js zur Bildvorverarbeitung und Anzeige
- Führt ONNX-Inferenz direkt im Browser mit `onnxruntime-web` aus
- Zeichnet erkannte Objekte farbig auf dem Bild ein

## Deployment
Für GitHub Pages genügt ein `push` in das Repository. Der GitHub Action-Workflow in `.github/workflows/publish.yml` rendert jetzt die Seite mit Quarto aus `index.qmd`.

1. Stelle sicher, dass GitHub Pages auf den Branch `main` (oder den aktiven Branch) und das Stammverzeichnis `/` eingestellt ist.
2. Die Startseite ist `index.html`.

> Hinweis: Quarto erzeugt die finale HTML-Seite automatisch. Du musst `index.qmd` nicht manuell in HTML konvertieren.

> Hinweis: Für diese statische GitHub Page ist keine Quarto-Datei (`index.qmd`) erforderlich. Alles läuft direkt über die vorhandene `index.html`.

## Verwendung
1. Öffne die Seite in einem Browser.
2. Wähle ein Bild aus.
3. Klicke auf "Erkennung starten".

## Struktur
- `index.html` &rarr; statische Benutzeroberfläche für die Bild- und Objekterkennung
- `models/yolov5n.onnx` &rarr; kleines YOLOv5-ONNX-Modell für Browserinferenz

## Hinweise
- Es ist keine Python- oder Flask-Serverlogik mehr erforderlich.
- Alles läuft clientseitig im Browser.

