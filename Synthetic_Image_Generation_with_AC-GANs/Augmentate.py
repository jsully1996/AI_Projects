import sys, os, re, traceback
from os.path import isfile
from multiprocessing.dummy import Pool
from counter import Counter
from ops.rotate import Rotate
from ops.fliph import FlipH
from ops.flipv import FlipV
from ops.zoom import Zoom
from ops.blur import Blur
from ops.noise import Noise
from ops.translate import Translate
from skimage.io import imread, imsave
from multiprocessing.dummy import Lock

class Counter:
    def __init__(self):
        self.lock = Lock()
        self._processed = 0
        self._error = 0
        self._skipped_no_match = 0
        self._skipped_augmented = 0

    def processed(self):
        with self.lock:
            self._processed += 1

    def error(self):
        with self.lock:
            self._error += 1

    def skipped_no_match(self):
        with self.lock:
            self._skipped_no_match += 1

    def skipped_augmented(self):
        with self.lock:
            self._skipped_augmented += 1

    def get(self):
        with self.lock:
            return {'processed' : self._processed, 'error' : self._error, 'skipped_no_match' : self._skipped_no_match, 'skipped_augmented' : self._skipped_augmented}

class Augment(object):
    def __init__(self):
        self.EXTENSIONS = ['png', 'jpg', 'jpeg', 'bmp']
        self.WORKER_COUNT = max(os.cpu_count() - 1, 1)
        self.OPERATIONS = [Rotate, FlipH, FlipV, Translate, Noise, Zoom, Blur]
        self.AUGMENTED_FILE_REGEX = re.compile('^.*(__.+)+\\.[^\\.]+$')
        self.EXTENSION_REGEX = re.compile('|'.join(['.*\\.' + n + '$' for n in self.EXTENSIONS]), re.IGNORECASE)
        self.thread_pool = None
        self.counter = None

    def build_augmented_file_name(original_name, ops):
        root, ext = os.path.splitext(original_name)
        result = root
        for op in ops:
            result += '__' + op.code
        return result + ext
    
    def work(d, f, op_lists):
        try:
            in_path = os.path.join(d,f)
            for op_list in op_lists:
                out_file_name = build_augmented_file_name(f, op_list)
                if isfile(os.path.join(d,out_file_name)):
                    continue
                img = imread(in_path)
                for op in op_list:
                    img = op.process(img)
                imsave(os.path.join(d, out_file_name), img)
    
            counter.processed()
        except:
            traceback.print_exc(file=sys.stdout)
    
    def process(dir, file, op_lists):
        thread_pool.apply_async(work, (dir, file, op_lists))
    
    if __name__ == '__main__':
        if len(sys.argv) < 3:
            print('Usage: {} <image directory> <operation> (<operation> ...)'.format(sys.argv[0]))
            sys.exit(1)
    
        image_dir = sys.argv[1]
        if not os.path.isdir(image_dir):
            print('Invalid image directory: {}'.format(image_dir))
            sys.exit(2)
    
        op_codes = sys.argv[2:]
        op_lists = []
        for op_code_list in op_codes:
            op_list = []
            for op_code in op_code_list.split(','):
                op = None
                for op in OPERATIONS:
                    op = op.match_code(op_code)
                    if op:
                        op_list.append(op)
                        break
    
                if not op:
                    print('Unknown operation {}'.format(op_code))
                    sys.exit(3)
            op_lists.append(op_list)
    
        counter = Counter()
        thread_pool = Pool(WORKER_COUNT)
        print('Thread pool initialised with {} worker{}'.format(WORKER_COUNT, '' if WORKER_COUNT == 1 else 's'))
    
        matches = []
        for dir_info in os.walk(image_dir):
            dir_name, _, file_names = dir_info
            print('Processing {}...'.format(dir_name))
    
            for file_name in file_names:
                if EXTENSION_REGEX.match(file_name):
                    if AUGMENTED_FILE_REGEX.match(file_name):
                        counter.skipped_augmented()
                    else:
                        process(dir_name, file_name, op_lists)
                else:
                    counter.skipped_no_match()
    
        print("Waiting for workers to complete...")
        thread_pool.close()
        thread_pool.join()
    
        print(counter.get())