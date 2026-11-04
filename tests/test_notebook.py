import json
from pathlib import Path


def test_original_notebook_preserved_without_large_outputs():
    notebook = Path(__file__).resolve().parents[1] / 'notebooks' / 'original_wattention.ipynb'
    data = json.loads(notebook.read_text())
    assert len(data['cells']) == 30
    assert all(not c.get('outputs') for c in data['cells'])
    sources = '\n'.join(''.join(c.get('source', [])) for c in data['cells'])
    assert 'class ElectricalModel' in sources
    assert 'class SetTransformerExtractor' in sources
    assert 'class LIBPackEnv' in sources
