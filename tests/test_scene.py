import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from scene import CLASSES, make_splits, validate_manifest

class SplitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for label in CLASSES:
            (self.root / label).mkdir()
            for i in range(10):
                (self.root / label / (str(i) + '.jpg')).write_bytes(f'{label}-{i}'.encode())

    def tearDown(self):
        self.tmp.cleanup()

    def test_determinism_disjointness_and_class_coverage(self):
        manifest = make_splits(self.root)
        self.assertEqual(manifest, make_splits(self.root))
        self.assertNotEqual(manifest, make_splits(self.root, 321))
        self.assertEqual([len(v) for v in manifest['splits'].values()], [48, 6, 6])
        validate_manifest(manifest, self.root)

    def test_duplicates_and_conflicting_labels(self):
        (self.root / 'forest' / 'copy.jpg').write_bytes(b'forest-0')
        manifest = make_splits(self.root)
        self.assertEqual(manifest['duplicates_removed'], 1)
        validate_manifest(manifest, self.root)
        (self.root / 'sea' / 'conflict.jpg').write_bytes(b'forest-0')
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            make_splits(self.root)

    def test_changed_content_and_overlapping_manifest_rejected(self):
        manifest = make_splits(self.root)
        overlapping = copy.deepcopy(manifest)
        overlapping['splits']['test'][0] = overlapping['splits']['train'][0]
        with self.assertRaises(ValueError):
            validate_manifest(overlapping, self.root)
        (self.root / manifest['splits']['test'][0]['path']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'changed'):
            validate_manifest(manifest, self.root)

    def test_notebook_top_to_bottom_without_training(self):
        notebook = json.loads((Path(__file__).resolve().parents[1] / 'Assignment3.ipynb').read_text(encoding='utf-8'))
        namespace = {}
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                self.assertEqual(cell['outputs'], [])
                self.assertIsNone(cell['execution_count'])
                exec(compile(''.join(cell['source']), '<notebook>', 'exec'), namespace)
        self.assertFalse(namespace['RUN_TRAINING'])
        self.assertFalse(namespace['RUN_EVALUATION'])

@unittest.skipUnless(importlib.util.find_spec('tensorflow'), 'TensorFlow not installed')
class ModelTests(unittest.TestCase):
    def test_resnet_preprocessing_and_custom_forward(self):
        import numpy as np
        import tensorflow as tf
        from scene import build_model
        raw = np.asarray([0., 127.5, 255.], dtype='float32')
        expected = tf.keras.applications.resnet_v2.preprocess_input(raw.copy())
        model = build_model('resnet', weights=None)  # no download, no fit
        rescale = next(layer for layer in model.layers if isinstance(layer, tf.keras.layers.Rescaling))
        np.testing.assert_allclose(rescale(raw).numpy(), expected)
        self.assertEqual(model.output_shape, (None, 6))
        self.assertFalse(next(layer for layer in model.layers if isinstance(layer, tf.keras.Model)).trainable)
        tf.keras.backend.clear_session()
        custom = build_model('custom')
        output = custom(np.zeros((1, 64, 64, 3), dtype='float32'), training=False).numpy()
        self.assertEqual(output.shape, (1, 6))
        np.testing.assert_allclose(output.sum(axis=1), 1, atol=1e-6)

    def test_synthetic_evaluation_writes_reports_and_plots(self):
        import tensorflow as tf
        from PIL import Image
        from scene import evaluate
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'data'
            run = Path(temporary) / 'run'
            run.mkdir()
            for label, name in enumerate(CLASSES):
                (root / name).mkdir(parents=True)
                for i in range(3):
                    Image.new('RGB', (8, 8), (label * 30, i * 50, 100)).save(root / name / f'{i}.png')
            manifest = make_splits(root)
            (run / 'splits.json').write_text(json.dumps(manifest))
            (run / 'config.json').write_text(json.dumps(dict(size=32)))
            inputs = tf.keras.Input((32, 32, 3))
            x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
            model = tf.keras.Model(inputs, tf.keras.layers.Dense(6, activation='softmax')(x))
            for kind in ('custom', 'resnet'):
                model.save(run / (kind + '.keras'))
                (run / (kind + '-history.json')).write_text(json.dumps(dict(accuracy=[.2], val_accuracy=[.2], loss=[2.], val_loss=[2.])))
            results = evaluate(root, run, 2)
            for kind in ('custom', 'resnet'):
                self.assertEqual(results[kind]['n_images'], 6)
                self.assertEqual(results[kind]['split'], 'test')
                self.assertEqual(len(results[kind]['confusion_matrix']), 6)
                self.assertTrue((run / (kind + '-confusion.png')).is_file())
                self.assertTrue((run / (kind + '-curves.png')).is_file())
            self.assertTrue((run / 'test-results.json').is_file())

if __name__ == '__main__':
    unittest.main()
