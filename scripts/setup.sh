git submodule update --init --recursive

python3 -m venv myenv
source myenv/bin/activate

pip install -e yolo_tracking
pip install -e .