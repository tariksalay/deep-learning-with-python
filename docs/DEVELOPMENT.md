# Development and validation

Use Python 3.11 or 3.12 in an isolated environment and install `requirements.txt`. Exercise folders and datasets retain their original paths. Change into the relevant exercise folder before running its script; many scripts use relative dataset filenames.

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Modernized Keras imports use `tensorflow.keras`, supported layers/utilities, current optimizer arguments, `accuracy` history names, and `argmax(model.predict(...))`. The LSTM grid-search exercise uses Keras' native scikit-learn wrapper with integer class labels. The glass-classification exercise reads its adjacent CSV and reports held-out rather than training accuracy. IDE metadata was removed from the tracked tree; it remains available in Git history.

Compatibility tests deliberately avoid running every coursework script: many scripts train models, download datasets/NLTK resources, open plots, expect absent local files, or require user input. Import checks and small model/layer construction smoke tests are separate from full experiment reproduction. Existing saved models, screenshots, and reported results are historical; no new accuracy claims are made. Some experiments still use test data as validation or fit preprocessing before splitting and need a separate experimental-design review before results are interpreted.
