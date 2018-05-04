import os

PNET_SETTINGS = {
    'path': os.path.join('mtcnn', 'dependencies', 'PNet', 'PNet-18'),
    'batch_size': 2048,
    'threshold': 0.9}
RNET_SETTINGS = {
    'path': os.path.join('mtcnn', 'dependencies', 'RNet', 'RNet-14'),
    'batch_size': 256,
    'threshold': 0.6}
ONET_SETTINGS = {
    'path': os.path.join('mtcnn', 'dependencies', 'ONet', 'ONet-16'),
    'batch_size': 16,
    'threshold': 0.7}
