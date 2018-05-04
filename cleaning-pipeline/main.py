import libraries
from utils import *
from image_converter import *
from parser import *

def main():
    # parse options
    opt = BaseOptions().parse()

    ## setting INPUT/OUTPUT dirs
    i_p = opt.i
    o_p = opt.o
    # ext = opt.e

    if opt.s:
        i_p = os.path.join(i_p, opt.s)
        o_p = os.path.join(o_p, opt.s)
    if opt.ss:
        i_p = os.path.join(i_p, opt.ss)
        o_p = os.path.join(o_p, opt.ss)
    print(i_p, o_p)

    if opt.conv:
        print(True)
        print('Converting images..')
        convert_images(i_p, o_p, opt.e, opt.sample_size)

if __name__ == "__main__":
    main()
