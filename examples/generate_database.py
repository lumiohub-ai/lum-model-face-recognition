from face_recognition import Database

creator = Database(device='gpu')
creator.generate(
    input_dir='data/images/newly-added-faces',
    output_file='data/embeddings/test_database.pkl'
)
