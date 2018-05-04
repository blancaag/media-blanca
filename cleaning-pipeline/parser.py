from libraries import *

class BaseOptions():
    def __init__(self):

        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('-i', type=str, required=True, help='input dir: path to dataset')
        self.parser.add_argument('-s', type=str, default=None, help='input subdir: name of input dir subdirs (e.g.: trainset) -same structure will be create din the ouput dir')
        self.parser.add_argument('-ss', type=str, default=None, help='subdir subdir: name of subdir within subdir -same structure will be create din the ouput dir')
        self.parser.add_argument('-o', type=str, default=None, help='output dir: if not provided it will be created within the input dir')
        self.parser.add_argument('-e', type=str, default='.jpg', help='extension: new image to convert the  <output dir>/landmarks')
        self.parser.add_argument('-im_size', type=int, default=None, help='image size: if -comp3D is enabled; image size of the output texture map')
        self.parser.add_argument('-sample_size', type=int, default=None, help='image size: if -comp3D is enabled; image size of the output texture map')

        self.parser.add_argument('-conv', action='store_true', help='convert: landmarks')

    def parse(self):
        if not self.initialized: self.initialize()
        self.opt = self.parser.parse_args()

        #setting INPUT/OUTPUT paths
        self.opt.i = os.path.join('../../datasets', self.opt.i)
        if not self.opt.o: self.opt.o = os.path.join(self.opt.i, 'output')

        self.initialized = True
        args = vars(self.opt); print(args)

        print('------------ Options -------------')
        for k, v in sorted(args.items()): print('%s: %s' % (str(k), str(v)))
        print('----------------------------------')

        return self.opt
