from face_recognition import Database

creator = Database(device='gpu')
creator.generate(
    input_dir='data/images/hb-facecrops',
    output_file='data/embeddings/hb-kor-latest.pkl'
)
