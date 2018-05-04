from libraries import *
from utils import *

def convert_images(i_path, o_path, ext, sample_size=None):
    """
    i_path: path to the images folder
    o_path: output path -by default create an 'isomaps' folder in 'i_path' [.obj and .isomap.png]
    sample_size: if desired to compute a sample of the images instead of all the 'i_path' folders
    """

    # supported image format
    i_formats = ['**/*.jpg', '**/*.JPG', '**/*.png']

    # setting paths
    i_p = os.path.join(o_path, 'preprocessed_labels')
    o_p = os.path.join(o_path, 'conversion')
    if not os.path.exists(o_p): os.makedirs(o_p)
    else: print('Warning: "%s" folder already exists: adding files..' %o_p)

    # images paths list
    i_pl = reduce(operator.add,
                  [glob(os.path.join(i_p, i), recursive=True) for i in i_formats],
                  [])
    if not sample_size: sample_size = len(i_pl)

    print('Total available images: %d' %len(i_pl))

    for i in i_pl[:sample_size]:
        st_0 = time() #
        # looping over the images
        f_name = i.split('/')[-1].split('.')[0]
        if f_name.split('_')[-1] == 'mirror': continue

        # skip if already exists
        if len(glob(os.path.join(o_p, f_name + ext))) > 0: # .obj / .mtl
            print(f_name, "already computed")
            continue

        print('Processing file: %s ' %i, end='')
        st_1 = time() #
        im = cv2.imread(i)
        cv2.imwrite(os.path.join(o_p, f_name + ext), im)

        print("---loop: %ss fitting: %ss---" % (time() - st_0, time() - st_1))

    n_im = len(glob(os.path.join(o_p, ext)))
    print('Total available images: %d | Total converted images: %d' %(len(i_pl), n_im))
