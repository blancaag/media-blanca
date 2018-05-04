import time
import torch
from test_internal import *
from utils.networks import *
from utils.datasets import *
import torch.optim as optim
from sklearn.metrics import log_loss, roc_auc_score
from torch.autograd import Variable
from torch.utils.data import DataLoader

sizes = [None, 64, 128, 256, 512]
for input_size in sizes:

    # parameters
    output_dir = 'output_multi_' + str(input_size)
    if not os.path.exists(output_dir): os.mkdir(output_dir)
    num_samples = 1048576 # 2^20 samples per epoch
    patch_size = 32
    batch_size = 256
    epochs = 100
    learning_rate = 1e-2
    weight_decay = 1e-4
    machine132 = True
    multiclass = True

    assert input_size in [None, 64, 128, 256, 512], "Input size must be one of: None, 64, 128, 256, 512"
    input_size = 'resized_' + str(input_size) + '_npy' if input_size is not None else 'context_0_npy'
    n_classes = 4 if multiclass else 2
    print 'learning rate: ' + str(learning_rate) + ' - patch size: ' + str(patch_size) + ' - batch size: ' + str(batch_size) + ' - total epochs: ' + str(epochs)
    print 'input folder: ' + str(input_size) + ' - output folder: ' + str(output_dir) + ' - multiclass: ' + str(multiclass)

    # filenames and labels
    if not machine132:
        dir_gen = '/media/dataserver/datasets/internal/face_spoofing/stable_applicant_selfie_2017_full/'+input_size+'/genuine_selfie'
        dir_pop = '/media/dataserver/datasets/internal/face_spoofing/stable_applicant_selfie_2017_full/'+input_size+'/picture_spoof'
        dir_pos = '/media/dataserver/datasets/internal/face_spoofing/stable_applicant_selfie_2017_full/'+input_size+'/screen_spoof'
        dir_pod = '/media/dataserver/datasets/internal/face_spoofing/stable_applicant_selfie_2017_full/'+input_size+'/document_spoof'
    else:
        dir_gen = '/home/gpuuser/Documents/stable_applicant_selfie_spoof_2017/passed_revolut_december/' + input_size
        dir_pop = '/home/gpuuser/Documents/stable_applicant_selfie_spoof_2017/warning_2017/'+input_size+'/picture_spoof'
        dir_pos = '/home/gpuuser/Documents/stable_applicant_selfie_spoof_2017/warning_2017/'+input_size+'/screen_spoof'
        dir_pod = '/home/gpuuser/Documents/stable_applicant_selfie_spoof_2017/warning_2017/'+input_size+'/document_spoof'
    print dir_gen
    print dir_pop 
    print dir_pos
    print dir_pod
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
    train_files, test_files = train_test_split([(dir_gen, 3), (dir_pop, 1), (dir_pos, 1), (dir_pod, 1)])
    print len(train_files), len(test_files)
    # load train data
    train_data = Internal(files=train_files, patch_size=patch_size, augment=True)
    train_sampler = train_data.get_weighted_random_sampler(num_samples) # equal genuine and spoof samples during training
    train_dataset = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, num_workers=11)
    # load val data
    val_data = Internal(files=test_files, patch_size=patch_size, augment=False)
    val_sampler = val_data.get_weighted_random_sampler(num_samples / 10) # more than just one patch per image for val
    val_dataset = DataLoader(val_data, batch_size=batch_size, sampler=val_sampler, num_workers=11)
    # create network
    network = ResNetCifar10(n_classes=n_classes).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    # print no. of trainable parameters
    model_parameters = filter(lambda p: p.requires_grad, network.parameters())
    print sum([np.prod(p.size()) for p in model_parameters])
    # run through all epochs
    train_loss = []
    val_loss = []
    val_acc2 = []
    val_acc = []
    val_auc = []
    for epoch in range(epochs):
        start = time.time()
        # train network
        network.train()
        network.train(True)
        running_loss = 0.0
        # lower learning rate
        if epoch in [75]:
            for param_group in optimizer.param_groups:
                learning_rate = learning_rate / 10
                param_group['lr'] = learning_rate
        for i, data in enumerate(train_dataset):
            # wrap data in Variables
            inputs, labels = data
            inputs = Variable(inputs.float()).cuda()
            labels = Variable(labels.long()).cuda()
            # forward pass
            outputs = network(inputs).view(-1, n_classes)
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # save statistics
            running_loss += loss.data[0]
        train_loss.append(running_loss / (i + 1))
        # test network
        network.eval()
        network.train(False)
        running_loss = 0.0
        y_pred = np.empty((0, n_classes))
        y_true = np.empty((0))
        for i, data in enumerate(val_dataset):
            # wrap data in Variables
            inputs, labels = data
            inputs = Variable(inputs.float(), volatile=True).cuda()
            labels = Variable(labels.long(), volatile=True).cuda()
            # forward pass
            outputs = network(inputs).view(-1, n_classes)
            loss = criterion(outputs, labels.long())
            # extra stats
            y_pred = np.append(y_pred, outputs.data.cpu().numpy(), axis=0)
            y_true = np.append(y_true, labels.data.cpu().numpy())
            # save statistics
            running_loss += loss.data[0]
        p2 = np.exp(y_pred) / np.exp(y_pred).sum(1, keepdims=True)
        val_loss.append(log_loss(y_true, p2))
        val_acc.append(sum(np.argmax(y_pred, 1)==np.array(y_true)) / float(len(y_true)))
        val_acc2.append(sum((np.argmax(y_pred, 1)>0)==(np.array(y_true)>0)) / float(len(y_true)))
        val_auc.append(roc_auc_score(y_true==0, y_pred[:,0]))
        print('[epoch %d] train loss: %.4f  val loss: %.4f  val acc: %.4f  auc: %.4f (%.2f sec)' %
                (epoch, train_loss[-1], val_loss[-1], val_acc[-1], val_auc[-1], time.time() - start))
    # test network on full images
    y_true, y_pred, y_live = test_network(network, test_files)
    # save network and results
    np.save(os.path.join(output_dir, 'y_true'), y_true)
    np.save(os.path.join(output_dir, 'y_pred'), y_pred)
    np.save(os.path.join(output_dir, 'y_live'), y_live)
    torch.save(network, os.path.join(output_dir, 'network'))
    np.save(os.path.join(output_dir, 'train_loss'), train_loss)
    np.save(os.path.join(output_dir, 'val_loss'), val_loss)
    np.save(os.path.join(output_dir, 'val_acc'), val_acc)
    np.save(os.path.join(output_dir, 'val_acc2'), val_acc2)
