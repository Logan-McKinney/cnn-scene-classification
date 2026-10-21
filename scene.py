"""Reproducible scene comparison. No training occurs on import."""
import argparse
import hashlib
import json
from pathlib import Path
import random

CLASSES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}

def make_splits(directory, seed=123):
    """Stratified 70/15/15 by unique content, independent of tf.data iteration."""
    root = Path(directory).resolve()
    if not root.is_dir() or sorted(p.name for p in root.iterdir() if p.is_dir()) != CLASSES:
        raise ValueError('Point to the directory containing exactly the six class folders')
    rng, seen = random.Random(seed), {}
    manifest = dict(schema=1, seed=seed, classes=CLASSES, duplicates_removed=0,
                    splits={'train': [], 'validation': [], 'test': []})
    for label, name in enumerate(CLASSES):
        records = []
        for path in sorted((root / name).rglob('*')):
            if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                continue
            if not path.resolve().is_relative_to(root):
                raise ValueError('Dataset symlink escapes root')
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in seen:
                if seen[digest] != label:
                    raise ValueError('Identical content has conflicting class labels')
                manifest['duplicates_removed'] += 1
                continue
            seen[digest] = label
            records.append(dict(path=path.relative_to(root).as_posix(), label=label, sha256=digest))
        if len(records) < 3:
            raise ValueError('Each class needs at least three distinct images')
        rng.shuffle(records)
        n_validation = n_test = max(1, int(len(records) * .15))
        manifest['splits']['test'].extend(records[:n_test])
        manifest['splits']['validation'].extend(records[n_test:n_test+n_validation])
        manifest['splits']['train'].extend(records[n_test+n_validation:])
    return manifest

def validate_manifest(manifest, root):
    root, seen_paths, seen_hashes = Path(root).resolve(), set(), set()
    if manifest['schema'] != 1 or manifest['classes'] != CLASSES:
        raise ValueError('Unsupported manifest')
    if set(manifest['splits']) != {'train', 'validation', 'test'}:
        raise ValueError('Expected train/validation/test splits')
    for records in manifest['splits'].values():
        if not records or {r['label'] for r in records} != set(range(len(CLASSES))):
            raise ValueError('Every split must contain all six classes')
        for row in records:
            path = (root / row['path']).resolve()
            if not path.is_relative_to(root) or row['path'] in seen_paths or row['sha256'] in seen_hashes:
                raise ValueError('Unsafe path or overlapping split content')
            if Path(row['path']).parts[0] != CLASSES[row['label']]:
                raise ValueError('Class label/path mismatch')
            if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
                raise ValueError('Dataset content changed since manifest creation')
            seen_paths.add(row['path'])
            seen_hashes.add(row['sha256'])

def dataset(records, root, size, batch, training=False, seed=123):
    import tensorflow as tf
    paths = [str(Path(root) / r['path']) for r in records]
    ds = tf.data.Dataset.from_tensor_slices((paths, [r['label'] for r in records]))
    if training:
        ds = ds.shuffle(len(records), seed=seed, reshuffle_each_iteration=True)
    def read(path, label):
        image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        image.set_shape((None, None, 3))
        return tf.image.resize(tf.cast(image, tf.float32), (size, size)), label
    # Both models accept raw RGB float pixels [0,255]; preprocessing lives in models.
    return ds.map(read, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True).batch(batch).prefetch(tf.data.AUTOTUNE)

def build_model(kind, size=64, seed=123, weights='imagenet'):
    import tensorflow as tf
    layers = tf.keras.layers
    inputs = tf.keras.Input((size, size, 3))
    x = layers.RandomFlip('horizontal', seed=seed)(inputs)
    x = layers.RandomRotation(.1, seed=seed + 1)(x)
    if kind == 'custom':
        x = layers.Rescaling(1. / 255)(x)
        for filters in (32, 64, 128):
            x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
            x = layers.MaxPooling2D()(x)
            x = layers.BatchNormalization()(x)
        x = layers.Flatten()(x)
        learning_rate = .001
    elif kind == 'resnet':
        # Equivalent to resnet_v2.preprocess_input on raw [0,255] RGB.
        # Serializable layer avoids unsafe Lambda deserialization.
        x = layers.Rescaling(1. / 127.5, offset=-1)(x)
        base = tf.keras.applications.ResNet50V2(weights=weights, include_top=False,
                                               input_shape=(size, size, 3))
        base.trainable = False
        x = base(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        learning_rate = .0001
    else:
        raise ValueError('Model must be custom or resnet')
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(.5)(x)
    model = tf.keras.Model(inputs, layers.Dense(len(CLASSES), activation='softmax')(x))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train(directory, output='runs/comparison', seed=123, epochs=15, size=64, batch=32):
    import tensorflow as tf
    if epochs < 1 or batch < 1 or size < 32:
        raise ValueError('epochs/batch must be positive; size must be at least 32')
    output = Path(output)
    if output.exists():
        raise ValueError('Use a fresh output directory to preserve prior experiments')
    manifest = make_splits(directory, seed)
    validate_manifest(manifest, directory)
    output.mkdir(parents=True)
    (output / 'splits.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    config = dict(seed=seed, epochs=epochs, size=size, batch=batch, tensorflow=tf.__version__,
                  classes=CLASSES, split='stratified unique content 70/15/15', preprocessing='raw-rgb-model-rescaling-v1')
    (output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    tf.config.experimental.enable_op_determinism()
    for kind in ('custom', 'resnet'):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)
        train_ds = dataset(manifest['splits']['train'], directory, size, batch, True, seed)
        val_ds = dataset(manifest['splits']['validation'], directory, size, batch)
        model = build_model(kind, size, seed)
        history = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3,
                                                                      restore_best_weights=True)])
        model.save(output / (kind + '.keras'))
        (output / (kind + '-history.json')).write_text(json.dumps(history.history, indent=2), encoding='utf-8')
    return output

def evaluate(directory, run_dir='runs/comparison', batch=32):
    """Use once after validation-based selection; never tune on this test set."""
    import tensorflow as tf
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, confusion_matrix
    run = Path(run_dir)
    manifest = json.loads((run / 'splits.json').read_text(encoding='utf-8'))
    config = json.loads((run / 'config.json').read_text(encoding='utf-8'))
    validate_manifest(manifest, directory)
    records = manifest['splits']['test']
    y_true = [r['label'] for r in records]
    results = {}
    for kind in ('custom', 'resnet'):
        model = tf.keras.models.load_model(run / (kind + '.keras'), safe_mode=True)
        predictions = model.predict(dataset(records, directory, config['size'], batch), verbose=0).argmax(axis=1)
        cm = confusion_matrix(y_true, predictions, labels=list(range(len(CLASSES))))
        results[kind] = dict(split='test', accuracy=float(accuracy_score(y_true, predictions)),
                             report=classification_report(y_true, predictions, labels=list(range(len(CLASSES))),
                                                          target_names=CLASSES, output_dict=True, zero_division=0),
                             confusion_matrix=cm.tolist(), n_images=len(records))
        display = ConfusionMatrixDisplay(cm, display_labels=CLASSES)
        display.plot(xticks_rotation=45, cmap='Blues')
        display.ax_.set_title(kind + ' - held-out test')
        display.figure_.tight_layout()
        display.figure_.savefig(run / (kind + '-confusion.png'))
        plt.close(display.figure_)
        history = json.loads((run / (kind + '-history.json')).read_text(encoding='utf-8'))
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, metric in zip(axes, ('accuracy', 'loss')):
            for key in (metric, 'val_' + metric):
                ax.plot(range(1, len(history[key]) + 1), history[key], label=key)
            ax.set(xlabel='Epoch', ylabel=metric, title=kind)
            ax.legend()
        fig.tight_layout()
        fig.savefig(run / (kind + '-curves.png'))
        plt.close(fig)
    (run / 'test-results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    return results

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['train', 'evaluate'])
    parser.add_argument('--data', required=True, help='Folder containing six class directories')
    parser.add_argument('--run', default='runs/comparison')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--size', type=int, default=64)
    parser.add_argument('--batch', type=int, default=32)
    args = parser.parse_args()
    if args.command == 'train':
        print(train(args.data, args.run, args.seed, args.epochs, args.size, args.batch))
    else:
        print(json.dumps(evaluate(args.data, args.run, args.batch), indent=2))

if __name__ == '__main__':
    main()
