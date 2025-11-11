\# ML Lottery Limits - Experimento Académico



\## Objetivo

Demostrar empíricamente que ningún algoritmo de ML puede predecir 

sorteos de lotería mejor que el azar.



\## Status

\- \*\*Fase:\*\* Recopilación inicial

\- \*\*Sorteos en BD:\*\* 2/500

\- \*\*Algoritmos:\*\* 2/15

\- \*\*Día:\*\* 1 de 540 (18 meses)



\## Próximos pasos

\- \[ ] Recopilar 10 sorteos históricos

\- \[ ] Implementar 3 algoritmos más

\- \[ ] Pre-registrar en OSF.io

\- \[ ] Configurar GitHub



\## Log

\- 2024-11-08: Proyecto iniciado ✅

\- 2024-11-08: Collector funcionando ✅

\- 2024-11-08: 2 algoritmos baseline ✅

```



3\. \*\*Crear archivo `.gitignore`\*\*

```

\# Python

venv/

\_\_pycache\_\_/

\*.pyc

\*.pyo

\*.egg-info/



\# Jupyter

.ipynb\_checkpoints/



\# Data (no subir sorteos por ahora)

data/raw/\*.json

data/processed/

data/predictions/



\# Results

results/



\# OS

.DS\_Store

Thumbs.db

