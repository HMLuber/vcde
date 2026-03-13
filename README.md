# Interaktives Website-Template
Dieses Repository dient als technische Grundlage für die Erstellung Ihrer Webseite. 
Die Vorlage integriert Quarto zur Dokumenterstellung, Python für computergestützte Analysen 
sowie beispielhaft Babylon.js für die Einbindung interaktiver 3D-Visualisierungen.

## Konfigurationsschritte (Setup)
Um eine eigene Arbeitsumgebung auf Basis dieser Vorlage zu erstellen, führen Sie bitte die folgenden Schritte strikt in der angegebenen Reihenfolge aus:

- **Repository forken**
Betätigen Sie die Schaltfläche "Fork" in der oberen rechten Ecke der GitHub-Oberfläche. Hierdurch wird eine identische Kopie des Projekts in Ihren persönlichen GitHub-Account übertragen.

- **Konfiguration der Workflow-Berechtigungen**
Standardmäßig sind Schreibzugriffe für automatisierte Prozesse in Forks deaktiviert. Zur Aktivierung der Website-Erstellung müssen Sie folgende Anpassungen vornehmen:

  - Navigieren Sie zu Settings > Actions > General.
    
  - Suchen Sie den Abschnitt Workflow permissions.

  - Aktivieren Sie die Option "Read and write permissions" und bestätigen Sie mit Save.

  - Wechseln Sie zum Reiter Actions und bestätigen Sie die Aktivierung der Workflows durch Klick auf "I understand my workflows, go ahead and enable them".

- **Aktivierung von GitHub Pages**
- 
  - Navigieren Sie zu Settings > Pages.

  - Wählen Sie unter dem Punkt "Build and deployment" bei Branch den Branch gh-pages aus.

  - Bestätigen Sie die Auswahl mit Save.
(Hinweis: Der Branch gh-pages wird erst nach dem ersten erfolgreichen Durchlauf der GitHub Action generiert, oder muss manuell erstellt werden).

## Workflow für Bearbeitung und Deployment
Sämtliche Änderungen an der Datei index.qmd führen nach einem Push zum Repository automatisch zur Ausführung der CI/CD-Pipeline:

- Python-Umgebung: Quarto führt enthaltene Code-Segmente aus und generiert die entsprechenden Abbildungen.

- Rendering: Die Markdown-Inhalte werden in ein statisches HTML-Dokument transformiert.

- Deployment: Die aktualisierte Website wird unter folgendem URL-Schema bereitgestellt: https://<ihr-username>.github.io/<repo-name>/

## Richtlinien zur Quarto-Syntax:
- Mathematische Formeln: Verwenden Sie die LaTeX-Notation, z. B. $E = mc^2$.

- Python-Berechnungen: Code-Blöcke müssen zwingend mit ```{python} eingeleitet werden.

- HTML/3D-Inhalte: Die Integration von Babylon.js-Skripten erfolgt innerhalb von ```{=html} Blöcken.

## Fehleranalyse (Troubleshooting)
- Fehlende Python-Grafiken: Überprüfen Sie das Protokoll unter dem Reiter "Actions" auf etwaige Installationsfehler der Bibliotheken matplotlib oder numpy.

- Inaktiver 3D-Canvas: Sollte die 3D-Umgebung nicht geladen werden, untersuchen Sie die Browser-Konsole (Taste F12) auf einen 404 (Not Found) Fehler. Dies deutet zumeist auf eine fehlerhafte Syntax im Verweis auf das Babylon.js-Framework hin.

