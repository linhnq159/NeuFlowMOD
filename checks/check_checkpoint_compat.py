"""Checkpoint loading checks for PyTorch 2.3 and newer."""

import pickle
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from tool.checkpoints import load_checkpoint


class UnsupportedCheckpointValue:
    pass


class CheckpointCompatibilityChecks(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / 'checkpoint.pth'

    def test_state_dict_and_training_checkpoint_without_safe_globals(self):
        tensor = torch.arange(6).reshape(2, 3)
        for checkpoint in ({'weight': tensor},
                           {'model': {'weight': tensor}, 'args': {'img_size': None},
                            'optimizer': {'state': {}, 'param_groups': []}, 'epoch': 3,
                            'rng_state': torch.get_rng_state()}):
            with self.subTest(training='model' in checkpoint):
                torch.save(checkpoint, self.path)
                with patch.object(torch.serialization, 'safe_globals', None, create=True):
                    loaded = load_checkpoint(self.path)
                state = loaded.get('model', loaded)
                torch.testing.assert_close(state['weight'], tensor)
                self.assertEqual(state['weight'].device.type, 'cpu')
                if 'model' in checkpoint:
                    self.assertIsNone(loaded['args']['img_size'])
                    self.assertEqual(loaded['epoch'], 3)
                    self.assertEqual(loaded['optimizer'], checkpoint['optimizer'])
                    torch.testing.assert_close(loaded['rng_state'], checkpoint['rng_state'])

    @unittest.skipUnless(hasattr(torch.serialization, 'safe_globals'),
                         'Namespace allowlisting requires PyTorch 2.4+')
    def test_namespace_metadata_with_safe_globals(self):
        torch.save({'model': {'weight': torch.ones(1)}, 'args': Namespace(img_size=[32, 32])}, self.path)
        self.assertEqual(load_checkpoint(self.path)['args'].img_size, [32, 32])

    def test_unsupported_objects_do_not_trigger_unrestricted_retry(self):
        torch.save({'unsupported': UnsupportedCheckpointValue()}, self.path)
        with patch.object(torch.serialization, 'safe_globals', None, create=True), \
                patch('tool.checkpoints.torch.load', wraps=torch.load) as loader:
            with self.assertRaises(pickle.UnpicklingError):
                load_checkpoint(self.path)
        loader.assert_called_once_with(self.path, map_location='cpu', weights_only=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
