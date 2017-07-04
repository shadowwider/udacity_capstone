import math
import os
import hashlib
from urllib2 import urlopen
import tarfile
import shutil
import pickle
import codecs

import numpy as np
from tqdm import tqdm
'''
lableDict = {'alt.atheism':[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 
                    'comp.graphics':[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     'comp.os.ms-windows.misc':1,
                     'comp.sys.ibm.pc.hardware':[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     'comp.sys.mac.hardware':[0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     'comp.windows.x':[0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                      'misc.forsale':[0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                       'rec.autos':[0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
                       'rec.motorcycles':[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                       'rec.sport.baseball':[0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
                       'rec.sport.hockey':2,
                        'sci.crypt':[0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                        'sci.electronics':[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
                        'sci.med':[0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
                         'sci.space':3,
                         'soc.religion.christian':4,
                         'talk.politics.guns':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
                         'talk.politics.mideast':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
                          'talk.politics.misc':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
                          'talk.religion.misc':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
                           }
'''

lableDict = {'alt.atheism':1, 
                    'comp.graphics':2,
                     'comp.os.ms-windows.misc':3,
                     'comp.sys.ibm.pc.hardware':4,
                     'comp.sys.mac.hardware':5,
                     'comp.windows.x':6,
                      'misc.forsale':7,
                       'rec.autos':8,
                       'rec.motorcycles':9,
                       'rec.sport.baseball':10,
                       'rec.sport.hockey':11,
                        'sci.crypt':12,
                        'sci.electronics':13,
                        'sci.med':14,
                         'sci.space':15,
                         'soc.religion.christian':16,
                         'talk.politics.guns':17,
                         'talk.politics.mideast':18,
                          'talk.politics.misc':19,
                          'talk.religion.misc':20
                           }
classDict = {
                     'comp.os.ms-windows.misc':1,
                   
                       'rec.sport.hockey':2,
                       
                         'sci.space':3,
                         'soc.religion.christian':4
                         
                           }
def _read32(bytestream):
    """
    Read 32-bit integer from bytesteam
    :param bytestream: A bytestream
    :return: 32-bit integer
    """
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def _unzip(save_path,extract_path, database_name, data_path):
    """
    Unzip wrapper with the same interface as _ungzip
    :param save_path: The path of the gzip files
    :param database_name: Name of database
    :param data_path: Path to extract to
    :param _: HACK - Used to have to same interface as _ungzip
    """
    print('Extracting {}...'.format(database_name))
    tar = tarfile.open(save_path, "r:gz")
    tar.extractall(extract_path)
    tar.close() 
    print('Extract down')   


def _ungzip(save_path, extract_path, database_name, _):
    """
    Unzip a gzip file and extract it to extract_path
    :param save_path: The path of the gzip files
    :param extract_path: The location to extract the data to
    :param database_name: Name of database
    :param _: HACK - Used to have to same interface as _unzip
    """
    # Get data from save_path
    with open(save_path, 'rb') as f:
        with gzip.GzipFile(fileobj=f) as bytestream:
            magic = _read32(bytestream)
            if magic != 2051:
                raise ValueError('Invalid magic number {} in file: {}'.format(magic, f.name))
            num_images = _read32(bytestream)
            rows = _read32(bytestream)
            cols = _read32(bytestream)
            buf = bytestream.read(rows * cols * num_images)
            data = np.frombuffer(buf, dtype=np.uint8)
            data = data.reshape(num_images, rows, cols)

    # Save data to extract_path
    for image_i, image in enumerate(
            tqdm(data, unit='File', unit_scale=True, miniters=1, desc='Extracting {}'.format(database_name))):
        Image.fromarray(image, 'L').save(os.path.join(extract_path, 'image_{}.jpg'.format(image_i)))


def get_image(image_path, width, height, mode):
    """
    Read image from image_path
    :param image_path: Path of image
    :param width: Width of image
    :param height: Height of image
    :param mode: Mode of image
    :return: Image data
    """
    image = Image.open(image_path)

    if image.size != (width, height):  # HACK - Check if image is from the CELEBA dataset
        # Remove most pixels that aren't part of a face
        face_width = face_height = 108
        j = (image.size[0] - face_width) // 2
        i = (image.size[1] - face_height) // 2
        image = image.crop([j, i, j + face_width, i + face_height])
        image = image.resize([width, height], Image.BILINEAR)

    return np.array(image.convert(mode))


def get_batch(image_files, width, height, mode):
    data_batch = np.array(
        [get_image(sample_file, width, height, mode) for sample_file in image_files]).astype(np.float32)

    # Make sure the images are in 4 dimensions
    if len(data_batch.shape) < 4:
        data_batch = data_batch.reshape(data_batch.shape + (1,))

    return data_batch


def images_square_grid(images, mode):
    """
    Save images as a square grid
    :param images: Images to be used for the grid
    :param mode: The mode to use for images
    :return: Image of images in a square grid
    """
    # Get maximum size for square grid of images
    save_size = math.floor(np.sqrt(images.shape[0]))

    # Scale to 0-255
    images = (((images - images.min()) * 255) / (images.max() - images.min())).astype(np.uint8)

    # Put images in a square arrangement
    images_in_square = np.reshape(
            images[:save_size*save_size],
            (save_size, save_size, images.shape[1], images.shape[2], images.shape[3]))
    if mode == 'L':
        images_in_square = np.squeeze(images_in_square, 4)

    # Combine images to grid image
    new_im = Image.new(mode, (images.shape[1] * save_size, images.shape[2] * save_size))
    for col_i, col_images in enumerate(images_in_square):
        for image_i, image in enumerate(col_images):
            im = Image.fromarray(image, mode)
            new_im.paste(im, (col_i * images.shape[1], image_i * images.shape[2]))

    return new_im


def download_extract(data_path):
    """
    Download and extract database
    :param database_name: Database name
    """
    
    database = "20news"
    url = 'http://www.qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz'        
    extract_path = os.path.join(data_path, 'data')
    save_path = os.path.join(data_path, '20news-bydate.tar.gz')
    extract_fn = _unzip
    

    if os.path.exists(extract_path):
        print('Found {} Data'.format(database))
        return
    else:
        os.makedirs(extract_path)   

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    if not os.path.exists(save_path):        
        raise 'data zip not find'

    
    #os.makedirs(extract_path)
    try:
        extract_fn(save_path, extract_path, database, data_path)
    except Exception as err:
        shutil.rmtree(extract_path)  # Remove extraction folder if there is an error
        raise err

    # Remove compressed data
    os.remove(save_path)

def read_file():
    sentences = {}
    path = {}
    collect = {}
    path['root'] = os.path.join('./', 'data')
    for tpart in ['20news-bydate-train', '20news-bydate-test']:
        path[tpart] = os.path.join(path['root'], tpart)
        sentences[tpart] = []
        folderList = os.listdir(path[tpart])
        collect[tpart]={}
        for folder in folderList:
            fileList = os.listdir(os.path.join(path[tpart], folder))
            i = 0    
            for eachf in fileList:
                fpath = os.path.join(path[tpart], folder, eachf)
                #print(fpath)
                with open(fpath, 'r') as f:
                    sentences[tpart].append((f.read(),folder))
                i += 1
            collect[tpart][folder] = i

    save_params(sentences)
    return sentences, collect

def tran_file():
    sentences = {}
    path = {}
    path['root'] = os.path.join('./', 'data')
    for tpart in ['20news-bydate-train', '20news-bydate-test']:
        path[tpart] = os.path.join(path['root'], tpart)
        sentences[tpart] = []
        folderList = os.listdir(path[tpart])
        for folder in folderList:
            fileList = os.listdir(os.path.join(path[tpart], folder))    
            for eachf in fileList:
                fpath = os.path.join(path[tpart], folder, eachf)
                print(fpath)
                tname = 't'+eachf
		topath = os.path.join(path[tpart], folder, tname)
		os.system('perl ss.pl '+fpath+' > '+topath)
    
	
def save_params(params):
    """
    Save parameters to file
    """
    pickle.dump(params, open('params.p', 'wb'))
	
def load_params():
    """
    Load parameters from file
    """
    return pickle.load(open('params.p', mode='rb'))


def choiced_data(params):
    x_train = []
    y_train = []
    for i,v in enumerate(params):  
        if v[1] in ['soc.religion.christian','comp.os.ms-windows.misc','rec.sport.hockey','sci.space']:      
            temp = []
            temp.append(v[0])
            x_train.append(temp)
            y_train.append(classDict[v[1]])
    return x_train,y_train

def labeled_data(params):
    x_train = []
    y_train = []
    for i,v in enumerate(params):         
        temp = []
        temp.append(v[0])
        x_train.append(temp)
        y_train.append(lableDict[v[1]])
    return x_train,y_train

class Dataset(object):
    """
    Dataset
    """
    def __init__(self, dataset_name, data_files):
        """
        Initalize the class
        :param dataset_name: Database name
        :param data_files: List of files in the database
        """
        DATASET_CELEBA_NAME = 'celeba'
        DATASET_MNIST_NAME = 'mnist'
        IMAGE_WIDTH = 28
        IMAGE_HEIGHT = 28

        if dataset_name == DATASET_CELEBA_NAME:
            self.image_mode = 'RGB'
            image_channels = 3

        elif dataset_name == DATASET_MNIST_NAME:
            self.image_mode = 'L'
            image_channels = 1

        self.data_files = data_files
        self.shape = len(data_files), IMAGE_WIDTH, IMAGE_HEIGHT, image_channels

   
class DLProgress(tqdm):
    """
    Handle Progress Bar while Downloading
    """
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        """
        A hook function that will be called once on establishment of the network connection and
        once after each block read thereafter.
        :param block_num: A count of blocks transferred so far
        :param block_size: Block size in bytes
        :param total_size: The total size of the file. This may be -1 on older FTP servers which do not return
                            a file size in response to a retrieval request.
        """
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num
