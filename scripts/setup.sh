git submodule update --init --recursive

python3 -m venv myenv
source myenv/bin/activate

pip install -e modules/yolo_tracking
pip install -e modules/insightface

pip install -e .