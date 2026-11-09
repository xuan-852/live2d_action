import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools.audit_model import audit, references


class ModelAuditTests(unittest.TestCase):
    def test_motion_references_exclude_metadata(self):
        self.assertEqual(list(references({'Motions': {'Idle': [
            {'File': 'idle.motion3.json', 'Sound': 'voice.wav', 'FadeInTime': 0.5}
        ]}})), ['idle.motion3.json', 'voice.wav'])

    def test_hash_and_missing_reference_failures(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = b'original'
            (root / 'model.moc3').write_bytes(data)
            (root / 'manifest.json').write_text(json.dumps({'files': [{
                'variant': '.', 'path': 'model.moc3', 'bytes': len(data),
                'sha256': hashlib.sha256(data).hexdigest()}]}), encoding='utf-8')
            (root / 'model.model3.json').write_text(json.dumps({'FileReferences': {
                'Moc': 'model.moc3'}}), encoding='utf-8')
            self.assertEqual(audit(root)['errors'], [])
            (root / 'model.moc3').write_bytes(b'changed')
            self.assertIn('Integrity mismatch', audit(root)['errors'][0])
            (root / 'model.moc3').unlink()
            self.assertEqual(len(audit(root)['errors']), 2)


if __name__ == '__main__':
    unittest.main()
