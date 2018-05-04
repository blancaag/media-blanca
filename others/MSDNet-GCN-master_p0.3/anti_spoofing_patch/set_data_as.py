import os
import time
import torch
from .test_internal import heatmap, test_network
# from .utils.networks import *
from .utils.datasets import Internal
import torch.optim as optim
from sklearn.metrics import log_loss, roc_auc_score
from torch.autograd import Variable
from torch.utils.data import DataLoader


def set_dataloaders(args):
        sizes = [None, 32, 64, 128, 256, 512]
        sizes = [args.size]
        for input_size in sizes:

            # parameters
            # output_dir = 'output_multi_' + str(input_size)
            # if not os.path.exists(output_dir): os.mkdir(output_dir)
            num_samples = 1048576 # 2^20 samples per epoch
            patch_size = args.patch_size
            batch_size = args.batch_size
            # epochs = 1
            # learning_rate = 1e-2
            # weight_decay = 1e-4
            # machine132 = False
            
            multiclass = True if args.nclasses == 4 else False

            assert input_size in [None, 32, 64, 128, 256, 512], "Input size must be one of: None, 64, 128, 256, 512"
            input_size = 'resized_' + str(input_size) if input_size is not None else 'context_0_npy'
            n_classes = 4 if multiclass else 2
            print ('patch size: ' + str(patch_size) + ' - batch size: ' + str(batch_size))
            # print ('input folder: ' + str(input_size)) + ' - output folder: ' + str(output_dir) + ' - multiclass: ' + str(multiclass))
            
            print('HERE HERE HERE multiclass  |  ncalsses:', multiclass, args.nclasses)
            # filenames and labels
            # if not machine132:
            #     dir_gen = '/media/dataserver/datasets/internal/face_spoofing/stable_applicant_selfie_2017_full/'+input_size+'/genuine_selfie'
            #     dir_pop = '/media/dataserver/datasets/internal/face_spoofing/stable_applicant_selfie_2017_full/'+input_size+'/picture_attack'
            #     dir_pos = '/media/dataserver/datasets/internal/face_spoofing/stable_applicant_selfie_2017_full/'+input_size+'/screen_attack'
            #     dir_pod = '/media/dataserver/datasets/internal/face_spoofing/stable_applicant_selfie_2017_full/'+input_size+'/document_attack'
            # else:
            #     dir_gen = '/home/gpuuser/Documents/stable_applicant_selfie_spoof_2017/passed_revolut_december/' + input_size
            #     dir_pop = '/home/gpuuser/Documents/stable_applicant_selfie_spoof_2017/warning_2017/'+input_size+'/picture_spoof'
            #     dir_pos = '/home/gpuuser/Documents/stable_applicant_selfie_spoof_2017/warning_2017/'+input_size+'/screen_spoof'
            #     dir_pod = '/home/gpuuser/Documents/stable_applicant_selfie_spoof_2017/warning_2017/'+input_size+'/document_spoof'
            # print (dir_gen)
            # print (dir_pop)
            # print (dir_pos)
            # print (dir_pod)
            
            gpath = '/workspace/blanca/stable_applicant_selfie_2017_full/' 
            
            dir_gen = gpath+input_size+'/genuine_selfie'
            dir_pop = gpath+input_size+'/picture_attack'
            dir_pos = gpath+input_size+'/screen_attack'
            dir_pod = gpath+input_size+'/document_attack'
            
            dir_gen = [(os.path.join(dir_gen, f), 0) for f in sorted(os.listdir(dir_gen))]
            dir_pop = [(os.path.join(dir_pop, f), 1) for f in sorted(os.listdir(dir_pop))]
            dir_pos = [(os.path.join(dir_pos, f), 2 if multiclass else 1) for f in sorted(os.listdir(dir_pos))]
            dir_pod = [(os.path.join(dir_pod, f), 3 if multiclass else 1) for f in sorted(os.listdir(dir_pod))]

            def train_test_split(dir_list):
                train = []
                test = []
                for dir_x, w in dir_list:
                    length = len(dir_x)
                    train.extend(dir_x[:-150*w])
                    test.extend(dir_x[-150*w:])
                return train, test

            # select files for this split
            # random.random.seed(1984)
            train_files, test_files = train_test_split([(dir_gen, 3), (dir_pop, 1), (dir_pos, 1), (dir_pod, 1)])
            print (len(train_files), len(test_files))
            # print(train_files[0], test_files[0])
            
            # load train data
            train_data = Internal(files=train_files, patch_size=patch_size, augment=True)
            train_sampler = train_data.get_weighted_random_sampler(num_samples) 
            
            # equal genuine and spoof samples during training
            train_dataset = DataLoader(train_data, 
                                       batch_size=batch_size, 
                                       sampler=train_sampler,
                                       pin_memory=True,
                                       num_workers=args.workers) #, drop_last=True) pin_memory=True)
            
            # load val data
            val_data = Internal(files=test_files, patch_size=patch_size, augment=False)
            val_sampler = val_data.get_weighted_random_sampler(num_samples / 10) 
            
            # more than just one patch per image for val
            val_dataset = DataLoader(val_data, 
                                     batch_size=batch_size, 
                                     sampler=val_sampler,
                                     pin_memory=True,
                                     num_workers=args.workers) #, drop_last=True) pin_memory=True)

            return train_dataset, val_dataset
    
def test_dataloaders():
    print('Testing dataloaders')
    for epoch in range(epochs):
        start = time.time()
        for i, data in enumerate(train_dataset):
            # wrap data in Variables
            inputs, labels = data
            print(inputs.shape, labels.shape)
            inputs = Variable(inputs.float()).cuda()
            labels = Variable(labels.long()).cuda()
        
        for i, data in enumerate(val_dataset):
            # wrap data in Variables
            inputs, labels = data
            print(inputs.shape, labels.shape)
            inputs = Variable(inputs.float(), volatile=True).cuda()
            labels = Variable(labels.long(), volatile=True).cuda()
        end = time.end()
        print('Taken %f sec. to load 2 batches (train & val)' %(start - end))