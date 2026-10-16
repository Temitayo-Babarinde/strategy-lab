import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from strategy_lab.containment import run_containment_demo

class ContainmentTests(unittest.TestCase):
    def test_no_local_fallback(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(run_containment_demo()['ok'])

    def test_evidence_and_cleanup(self):
        client = MagicMock()
        sandbox = client.create.return_value
        sandbox.id = 'test-sandbox'
        sandbox.process.exec.return_value = SimpleNamespace(exit_code=0, result='PROBE_JSON:' + json.dumps(dict(secrets_absent=True, network_connected=False, network_outcome='TimeoutError')))
        with patch.dict(os.environ, {'DAYTONA_API_KEY': 'test'}), patch('daytona.Daytona', return_value=client):
            result = run_containment_demo()
        self.assertTrue(result['ok'])
        self.assertTrue(result['cleaned_up'])
        self.assertTrue(client.create.call_args.args[0].network_block_all)
        client.delete.assert_called_once_with(sandbox, wait=True, timeout=30)

    def test_cleanup_failure_withholds_success(self):
        client = MagicMock()
        client.create.return_value.process.exec.return_value = SimpleNamespace(exit_code=1, result='')
        client.delete.side_effect = RuntimeError('failure')
        with patch.dict(os.environ, {'DAYTONA_API_KEY': 'test'}), patch('daytona.Daytona', return_value=client):
            result = run_containment_demo()
        self.assertFalse(result['ok'])
        self.assertFalse(result['cleaned_up'])
