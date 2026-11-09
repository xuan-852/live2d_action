import json
from pathlib import Path
import tempfile
import unittest

from tools.check_capture import check


class CaptureTests(unittest.TestCase):
    def test_pairing_time_and_requested_state(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'complete.txt').write_text('done\n', encoding='utf-8')
            state = {'frame': 0, 'actionTime': 0, 'requested': [1], 'applied': [1],
                     'drawables': [1], 'parts': [1]}
            path = root / '0000.json'
            path.write_text(json.dumps(state), encoding='utf-8')
            (root / '0000.png').write_bytes(b'\x89PNG\r\n\x1a\n')
            self.assertTrue(check(root)['identicalImages'])
            state['actionTime'] = 0.1
            path.write_text(json.dumps(state), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Nonuniform'):
                check(root)
            state['actionTime'] = 0
            state['applied'] = [2]
            path.write_text(json.dumps(state), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'parameters differ'):
                check(root)


if __name__ == '__main__':
    unittest.main()
