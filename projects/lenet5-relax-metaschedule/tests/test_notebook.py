"""Static notebook checks without installing TVM or attaching GPU."""
import ast
import json
from pathlib import Path


def test_notebook_code_is_valid_python():
    path = Path(__file__).resolve().parents[1] / 'notebooks' / 'relax_metaschedule_colab.ipynb'
    nb = json.loads(path.read_text())
    assert nb['nbformat'] == 4
    code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
    assert len(code_cells) >= 6
    for i, c in enumerate(code_cells):
        ast.parse(''.join(c['source']), filename=f'{path}:cell-{i}')


def test_no_fabricated_results_in_notebook():
    path = Path(__file__).resolve().parents[1] / 'notebooks' / 'relax_metaschedule_colab.ipynb'
    nb = json.loads(path.read_text())
    code = '\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
    assert 'relax_meta.benchmark' in code
    assert '--trials' in code
    assert 'MetaSchedule' in path.read_text()
    assert not any(c['outputs'] for c in nb['cells'] if c['cell_type'] == 'code')
