# Python, Machine Learning, and Deep Learning Coursework

**Python · pandas · scikit-learn · TensorFlow/Keras · NLP · CNNs · RNNs**

A collection of separate coursework experiments tracing a path from Python fundamentals to classical machine learning, text processing, neural networks, and dimensionality reduction. Each exercise has its own data and execution assumptions.

## What this project demonstrates

- Classical and neural modeling workflows across several data types.
- Data preparation, feature extraction, and experiment comparison.
- Modernized API compatibility with explicit limits on full reproduction.

## Start here

| Read | Use it for |
| --- | --- |
| [Project walkthrough](docs/PROJECT_GUIDE.md) | Purpose, workflow, repository map, and a concrete example |
| [Code and artifact reference](docs/CODE_REFERENCE.md) | Important files, functions, classes, and their responsibilities |
| [Development guide](docs/DEVELOPMENT.md) | Setup, configuration, commands, tests, and remaining limitations |

## Explore locally

Use Python 3.11 or 3.12 in a virtual environment.

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Inspect an exercise’s input filenames, then change into its folder before running it. There is no single command that safely runs the whole collection.

## Validation and scope

Four tests cover TensorFlow imports, CNN construction/inference, an LSTM batch and sklearn wrapper, and phrase-preprocessing schemas. They do not reproduce every training run or validate historical accuracy. Several scripts prompt, download, train, write files, or open plots at module level.



---

[GitHub profile](https://github.com/tariksalay) · [Portfolio](https://tariksalay.github.io/) · [Earlier README](README_HISTORY.md)

The earlier README is preserved as historical context; the guides above describe this branch's implementation.
