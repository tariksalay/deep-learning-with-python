"""Exercise APIs without starting dataset downloads or full coursework training."""
import ast
from pathlib import Path
import unittest
import numpy as np
from sklearn.base import is_classifier

ROOT = Path(__file__).resolve().parents[1]


def execute(nodes, namespace, path):
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)


class CompatibilityTests(unittest.TestCase):
    def test_tensorflow_imports_resolve(self):
        for path in ROOT.rglob('*.py'):
            if 'tests' in path.parts:
                continue
            tree = ast.parse(path.read_text())
            imports = [node for node in tree.body if isinstance(node, ast.ImportFrom)
                       and (node.module or '').startswith('tensorflow')]
            execute(imports, {}, path)

    def test_cnn_builds_and_predicts_without_downloading_cifar(self):
        path = ROOT / 'M2-ICP4_ImageClassificationCNN/image_classification.py'
        namespace = {'X_train': np.zeros((2, 32, 32, 3)), 'num_classes': 10}
        tree = ast.parse(path.read_text())
        for node in tree.body:
            is_import = isinstance(node, (ast.Import, ast.ImportFrom))
            is_assignment = isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id in {'model', 'epochs', 'lrate', 'decay', 'sgd'}
                for target in node.targets)
            is_model_call = (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                             and isinstance(node.value.func, ast.Attribute)
                             and isinstance(node.value.func.value, ast.Name)
                             and node.value.func.value.id == 'model'
                             and node.value.func.attr in {'add', 'compile'})
            if is_import or is_assignment or is_model_call:
                execute([node], namespace, path)
        output = namespace['model'](np.zeros((1, 32, 32, 3)), training=False).numpy()
        self.assertEqual(output.shape, (1, 10))
        np.testing.assert_allclose(output.sum(axis=1), 1, atol=1e-6)

    def test_lstm_wrapper_is_classified_correctly_by_sklearn(self):
        path = ROOT / 'M2-ICP5_SentimentAnalysis+LSTM/sentimentAnalysis.py'
        tree = ast.parse(path.read_text())
        namespace = {'max_fatures': 2000, 'embed_dim': 128, 'lstm_out': 196}
        nodes = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
        execute(nodes, namespace, path)
        estimator = namespace['SKLearnClassifier'](model=namespace['createmodel'])
        self.assertTrue(is_classifier(estimator))
        estimator.set_params(fit_kwargs={'epochs': 1, 'batch_size': 2, 'verbose': 0})
        x = np.zeros((2, 4), dtype=np.int32)
        y = np.eye(2, dtype=np.float32)
        model = namespace['createmodel'](x, y)
        result = model.train_on_batch(x, y, return_dict=True)
        self.assertTrue(np.isfinite(result['loss']))


if __name__ == '__main__':
    unittest.main()
