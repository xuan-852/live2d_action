import io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from tools.control.opencode_model_client import OpenCodeModelClient,content_json


class ModelClientTests(unittest.TestCase):
    def client(self,root):
        p=Path(root)/'opencode.json'
        p.write_text(json.dumps(dict(model='a6api/test-model',provider=dict(a6api=dict(npm='@ai-sdk/openai-compatible',
            options=dict(baseURL='http://127.0.0.1:1234/v1',apiKey='fake-unit-test-key'),models={'test-model':{}})))),encoding='utf-8')
        return OpenCodeModelClient(p,provider='a6api',model='test-model')

    def request(self):return dict(requestId=3,observationTime=1.,reason='actual-execution-limited',conversation=[],scene={},actual={},lastExecution={},outputContract={})

    def test_response_binding_and_public_metadata_omit_credentials(self):
        with tempfile.TemporaryDirectory() as root:
            c=self.client(root);raw=json.dumps(dict(model='returned-model',choices=[dict(message=dict(content='{"id":"m"}',reasoning_content='private'),finish_reason='stop')]))
            with patch('tools.control.opencode_model_client.urlopen',return_value=io.StringIO(raw)) as call:
                reply,meta=c.respond(self.request())
            self.assertEqual(reply['requestId'],3);self.assertEqual(reply['observationTime'],1.)
            self.assertEqual(call.call_args.args[0].full_url,'http://127.0.0.1:1234/v1/chat/completions')
            persisted=json.dumps([meta,c.public_info()])
            self.assertNotIn('fake-unit-test-key',persisted);self.assertNotIn('private',persisted)
            self.assertEqual(meta['modelReturned'],'returned-model')

    def test_invalid_generated_json_is_preserved_for_validation_repair(self):
        with tempfile.TemporaryDirectory() as root:
            c=self.client(root);raw=json.dumps(dict(choices=[dict(message=dict(content='not JSON'),finish_reason='length')]))
            with patch('tools.control.opencode_model_client.urlopen',return_value=io.StringIO(raw)):
                reply,meta=c.respond(self.request())
            self.assertIn('invalidGeneratedContent',reply['plan']);self.assertEqual(meta['content'],'not JSON')

    def test_json_fence_is_not_a_license_for_prose_or_nonobject_output(self):
        self.assertEqual(content_json('```json\n{"id":"m"}\n```'),{'id':'m'})
        for text in ('[]','Here is my plan: {}'):
            with self.assertRaises(ValueError):content_json(text)
