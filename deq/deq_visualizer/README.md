# deq_visualizer

## Export to HTML

To share a Jupyter notebook that contains deq visualizer widgets, the best way is to export an HTML file.
The visualizer will be fully functional and self contained.
The export is only supported on browser using jupyter notebook or jupyter lab ([not possible on vscode](https://github.com/microsoft/vscode-jupyter/issues/4404) as of today).

0. Make sure you set `_DEV = False` in `deq.visual.widget` and build the frontend `npm run build`
1. Open the jupyter notebook in a browser
2. In the "Settings" menu, enable "Save Widget State Automatically"
3. Save the jupyter notebook (the size of the notebook should be >1MB)
4. run `jupyter nbconvert --to html <name>.ipynb`, which will output to `<name>.html`

Limitation: only the state set from the Python code are persistent. For example, if one set the `selected` field in Python code, the exported HTML will have the correspond element(s) selected. The selection of from web UI will not be persistent.
