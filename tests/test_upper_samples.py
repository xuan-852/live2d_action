import json,tempfile,unittest
from pathlib import Path
from tools.control.prepare_upper_samples import pair_request

class UpperSampleTests(unittest.TestCase):
    def test_pair_requires_id_and_seed(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'reference.json';p.write_text(json.dumps({'caption':'a','seed':2}))
            self.assertIsNone(pair_request(p,[{'id':'a','seed':1,'text':'wrong'}]))
            self.assertEqual(pair_request(p,[{'id':'a','seed':2,'text':'correct'}])['text'],'correct')
            with self.assertRaises(ValueError):pair_request(p,[{'id':'a','seed':2,'text':'one'},{'id':'a','seed':2,'text':'two'}])
