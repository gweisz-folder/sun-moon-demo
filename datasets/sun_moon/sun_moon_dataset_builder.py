"""sun_moon dataset."""

import os
from PIL import Image
import tensorflow_datasets as tfds


class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for sun_moon dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'image': tfds.features.Image(shape=(None, None, 3)),
                'filename': tfds.features.Text(),
                'label': tfds.features.ClassLabel(names=['sun', 'moon']),
            }),
            supervised_keys=('image', 'label'),
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        return {
            'train': self._generate_examples('train'),
            'validation': self._generate_examples('validation'),
        }

    def _generate_examples(self, path):
        """Yields examples."""
        for f in os.listdir(path):
            label = ''

            if f.startswith('sun-'):
                label = 'sun'
            elif f.startswith('moon-'):
                label = 'moon'
            else:
                raise Exception("Unrecognized label in filename")

            if not f.endswith('.jpg'):
                raise Exception("Unexpected file extension")

            with Image.open(os.path.join(path, f)) as image:
                yield f, {
                    'image': image,
                    'filename': f,
                    'label': label,
                }
