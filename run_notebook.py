"""Execute the molecular prediction notebook with the active Python environment."""

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter


def main():
    root = Path(__file__).resolve().parent
    os.chdir(root)
    start = time.monotonic()
    notebook = nbformat.read(root / 'blood_brain_barrier.ipynb', as_version=4)
    # A temporary kernel uses this interpreter rather than a global Jupyter kernel.
    with tempfile.TemporaryDirectory(prefix='bbbp-kernel-') as directory:
        kernel = Path(directory) / 'kernels' / 'bbbp-local'
        kernel.mkdir(parents=True)
        (kernel / 'kernel.json').write_text(json.dumps({
            'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
            'display_name': 'Molecular prediction', 'language': 'python',
        }))
        previous = os.environ.get('JUPYTER_PATH')
        os.environ['JUPYTER_PATH'] = directory + (os.pathsep + previous if previous else '')
        try:
            NotebookClient(notebook, timeout=7200, kernel_name='bbbp-local',
                           resources={'metadata': {'path': str(root)}}).execute()
        finally:
            if previous is None:
                os.environ.pop('JUPYTER_PATH', None)
            else:
                os.environ['JUPYTER_PATH'] = previous
    nbformat.write(notebook, root / 'blood_brain_barrier.ipynb')
    html, _ = HTMLExporter().from_notebook_node(notebook)
    (root / 'outputs' / 'blood_brain_barrier.html').write_text(html)
    print(f'Notebook completed in {(time.monotonic() - start) / 60:.2f} minutes.', flush=True)


if __name__ == '__main__':
    main()
