import os
import io
import PIL
import dlib
import torch
import requests
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.misc import imresize, imread
detector = dlib.get_frontal_face_detector()

SIZE = 128
PATCH_SIZE = 32
RESOLUTION = 32
MODEL_LOCATION = os.path.join('utils', 'network_patchnormlr3')

# load model and set to evaluation mode
model = torch.load(MODEL_LOCATION).cuda()
model.eval()
model.train(False)

def detect_face(img):
    ''' Detect faces in the image and return a crop of the largest face. '''
    # detect faces
    dets, conf, _ = detector.run(img)
    # filter detections with conf <= 0
    dets = [d for d, c in zip(dets,conf) if c > 0]
    # if at least one face was detected
    if len(dets) > 0:
        # calculate face sizes per detection
        size = [max(d.bottom(), 0) - max(d.top(), 0) + max(d.right(), 0) - max(d.left(), 0) for d in dets]
        # select detection corresponding with largest face
        det = dets[np.argmax(size)]
        # get face positions
        top, bot, left, right = max(det.top(), 0), max(det.bottom(), 0), max(det.left(), 0), max(det.right(), 0)
        # return cropped image
        return img[top:bot, left:right]
    return None

def create_heatmap(face_crop):
    ''' Select uniformly sampled patches from the face_crop and run them through the network. '''
    face_crop = face_crop.transpose(2, 0, 1)
    c, h, w = face_crop.shape
    # create a batch with uniformly sampled patches
    hr = float(h - PATCH_SIZE) / (RESOLUTION - 1)
    wr = float(w - PATCH_SIZE) / (RESOLUTION - 1)
    batch = np.zeros((RESOLUTION**2, 3, PATCH_SIZE, PATCH_SIZE))
    for y in range(RESOLUTION):
        for x in range(RESOLUTION):
            patch = face_crop[:, int(y * hr):int(y * hr + PATCH_SIZE), int(x * wr):int(x * wr + PATCH_SIZE)]
            patch = (patch - np.mean(patch)) / (np.std(patch) + 1e-8)
            batch[y * RESOLUTION + x, :, :, :] = patch
    # network makes a prediction per patch
    inputs = Variable(torch.from_numpy(batch).float(), volatile=True).cuda()
    #outputs = F.softmax(model(inputs), dim=1).data.cpu().numpy()
    #score = np.median(outputs, 0)[0]
    outputs = model(inputs).data.cpu().numpy()
    score = np.mean(np.clip(outputs, np.percentile(outputs, 5, 0), np.percentile(outputs, 95, 0)), 0)
    score = np.exp(score)[0] / sum(np.exp(score)) 
    return score
    #return outputs.reshape((RESOLUTION, RESOLUTION, -1))

def predict(photo_id):
    ''' Turns input image into a spoof attempt prediction. '''
    # 1. download image
    url = 'https://imago.onfido.com/api/{}/{}/download'.format('live_photos', str(photo_id))
    im_response = requests.get(url)
    try:
        image = imread(io.BytesIO(im_response.content))
    except:
        return 'failed: reading image'
    # filter greyscale images
    if len(image.shape) is not 3 or image.shape[2] < 3:
        return 'failed: image shape'
    # remove possible alpha channel
    if image.shape[2] > 3: 
        image = image[:,:,:3]
    # get exit and rotate
    img = PIL.Image.open(io.BytesIO(im_response.content))
    if hasattr(img, '_getexif') and img._getexif() is not None and 274 in img._getexif():
        orientation = img._getexif()[274]
        if orientation in [3,4]:
            image = np.rot90(image, 2)
        if orientation in [5,6]:
            image = np.rot90(image, 3)
        if orientation in [7,8]:
            image = np.rot90(image, 1)
    # detect face in image
    face_crop = detect_face(image)
    if face_crop is None:
        return 'failed: face detection'
    # resize image
    face_crop = imresize(face_crop, (SIZE, SIZE))
    # spoof prediction
    heatmap = create_heatmap(face_crop)
    return heatmap
    # transform heatmap into a spoof prediction (1.0 is spoof attack)
    #spoof_score = np.mean(heatmap[:, :, 0])
    return spoof_score
