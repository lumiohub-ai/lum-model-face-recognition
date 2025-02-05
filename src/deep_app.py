from deepface import DeepFace
from pathlib import Path



info_deep_face = DeepFace.find(
            img_path = 'Bahodir.jpg',
            db_path = "data/images",
            )
values = info_deep_face[0]._values
matched_path = Path(values[0][0])
name = matched_path.stem

print(name)
print(values)