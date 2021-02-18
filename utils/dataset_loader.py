import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torch
import h5py

from eval import get_mask_from_bbox


class BasicDataset(Dataset):
    TARGET_IMG_PATH = "target_image_path"
    MASK_IMG_PATH = "mask_image_path"
    BBOX_PATH = "bbox_path"
    QUERY_FULL_IMG_PATH = "target_image_bbox_path"

    # TODO: Check if the values are empty
    def __init__(self, imgs_dir: str, masks_dir: str, dataset_name: str, mask_img_dim: int = 256, query_dim: int = 64,
                 bbox_suffix: str = '.bboxes.txt', save_to_disk: bool = False, skip_bbox_lines: int = 0):
        assert imgs_dir is not None and imgs_dir is not "", 'Please insert a directory for the images'
        assert masks_dir is not None and masks_dir is not "", 'Please insert a directory for the masks'
        assert dataset_name is not None and dataset_name is not "", 'Please insert the name of the dataset'
        assert mask_img_dim > 1, 'The dimension of mask and image must be higher than 1'
        assert query_dim > 1, 'The dimension of query image must be higher than 1'

        self.imgs_dir = fix_input_dir(imgs_dir)
        self.masks_dir = fix_input_dir(masks_dir)
        self.processed_img_dir = str(self.imgs_dir[:self.imgs_dir.rindex(os.path.sep) + 1]) + "preprocessed"
        self.mask_img_dim = mask_img_dim
        self.query_dim = query_dim
        self.bbox_suffix = bbox_suffix
        self.save_to_disk = save_to_disk
        self.skip_bbox_lines = skip_bbox_lines

        assert os.path.isdir(self.imgs_dir), f"Bad path for images directory: {self.imgs_dir}"
        assert os.path.isdir(self.masks_dir), f"Bad path for masks directory: {self.masks_dir}"

        if save_to_disk:
            # create processed image's directory, if not exists yet
            try:
                os.mkdir(self.processed_img_dir)
            except FileExistsError:
                # some previous instance generate this directory, no need to raise an exception
                pass

        # List of dict. Every dict refers to an image with 4 keys:
        #       target, mask, bbox, query_target_bbox
        self.imgs_path = []

        if "FlickrLogos" in dataset_name:
            self.flickrlogos32_load()
        elif "TopLogos-10" in dataset_name:
            self.toplogos10_load()
        print(f"You have {len(self.imgs_path)} triplets")

    def __len__(self):
        return len(self.imgs_path)

    def toplogos10_load(self):

        # get bbox path
        bbox_path = None
        for bbox_paths, _, bbox_list in os.walk(self.masks_dir):
            for bbox_file in bbox_list:
                if self.bbox_suffix in bbox_file:
                    bbox_path = get_class_file_path(bbox_paths, bbox_file)
                    break
            if bbox_path:
                break

        # get query image path
        query_full_img_path = f"{bbox_path[:bbox_path.index(self.bbox_suffix)]}"

        # get target images path and fill "self.images_path"
        for target_imgs_paths, _, target_imgs_list in os.walk(self.imgs_dir):
            for target_img_name in target_imgs_list:
                target_img_root_path, target_img_extension = os.path.splitext(
                    os.path.join(target_imgs_paths, target_img_name))
                if target_img_extension == ".jpg":
                    self.imgs_path.append(
                        {self.TARGET_IMG_PATH: get_class_file_path(target_imgs_paths, target_img_name),
                         self.MASK_IMG_PATH: f"{get_class_file_path(target_imgs_paths, target_img_name)}{self.bbox_suffix}",
                         self.BBOX_PATH: bbox_path,
                         self.QUERY_FULL_IMG_PATH: query_full_img_path})

    def flickrlogos32_load(self):

        # dict with merged masks path
        # key = target image file name
        # value = merged mask's file path
        masks_dict = {}

        # put stuff into masks_dict
        for masks_paths, _, masks_files in os.walk(self.masks_dir):
            for mask_file_name in masks_files:
                _, mask_extension = os.path.splitext(os.path.join(masks_paths, mask_file_name))
                if mask_extension == ".png" and "merged" in mask_file_name:
                    masks_dict[mask_file_name[:mask_file_name.rindex(".mask")]] = get_class_file_path(masks_paths,
                                                                                                      mask_file_name)

        # dict with every image of every class
        # key = class name
        # value = dict with images type and path
        #       key = type of image (target, query, mask)
        #       value = path of the file
        img_path_element = {}

        # put stuff into img_path_element
        for target_imgs_paths, _, target_imgs_files in os.walk(self.imgs_dir):
            target_img_class = target_imgs_paths[target_imgs_paths.rindex(os.path.sep) + 1:]
            for target_img_name in target_imgs_files:
                target_img_root_path, target_img_extension = os.path.splitext(
                    os.path.join(target_imgs_paths, target_img_name))
                if target_img_extension == ".jpg" and "no-logo" not in target_img_root_path:
                    x = {self.TARGET_IMG_PATH: get_class_file_path(target_imgs_paths, target_img_name),
                         self.MASK_IMG_PATH: masks_dict[target_img_name],
                         self.BBOX_PATH: f'{masks_dict[target_img_name][:masks_dict[target_img_name].rindex(".mask")]}{self.bbox_suffix}'}
                    try:
                        img_path_element[target_img_class].append(x)
                    except KeyError:
                        img_path_element[target_img_class] = [x]

        # fill "imgs_path" variable. it generate every couple (target image, query image) for the same class
        # for now, it only skips couple (target, query) of the same image
        for target_img_class in img_path_element:
            items_class = img_path_element[target_img_class]
            for outer_img in items_class:
                bbox_path = outer_img[self.BBOX_PATH]
                outer_target_img_path = outer_img[self.TARGET_IMG_PATH]
                for inner_img in items_class:
                    if not outer_img == inner_img:
                        target_img_path = inner_img[self.TARGET_IMG_PATH]
                        mask_img_path = inner_img[self.MASK_IMG_PATH]
                        self.imgs_path.append({self.TARGET_IMG_PATH: target_img_path,
                                               self.MASK_IMG_PATH: mask_img_path,
                                               self.BBOX_PATH: bbox_path,
                                               self.QUERY_FULL_IMG_PATH: outer_target_img_path})

    # preprocess the images. then save in file and return a list triplet [query image, target image, mask image]. how?
    # stretch the target image
    # stretch, crop and stretch again the query image
    # stretch the mask image
    # TODO: Check if the values are empty
    @classmethod
    def preprocess(cls, target_img_path: str, bbox_path: str, query_full_img_path: str, skip_bbox_lines: int = 0,
                   img_dim: int = 256, query_img_dim: int = 64, mask_img_path: str = None) -> dict:
        # Target image #

        pil_target_img = Image.open(target_img_path)
        # stretch the image
        pil_resized_target_img = pil_target_img.resize((img_dim, img_dim))

        # Query image #

        pil_target_img_bbox = Image.open(query_full_img_path)
        pil_resized_target_img_bbox = pil_target_img_bbox.resize((img_dim, img_dim))

        # we will resize, crop and resize again the image but we have the coordinates of the non resized bounding box
        target_img_width, target_img_height = pil_target_img_bbox.size
        resized_target_img_width, resized_target_img_height = pil_resized_target_img_bbox.size
        percent_width = round(100 * int(resized_target_img_width) / (int(target_img_width)), 2) / 100
        percent_height = round(100 * int(resized_target_img_height) / (int(target_img_height)), 2) / 100

        # open the bounding box file
        with open(bbox_path) as bbox_file:
            # read only the first line of the bbox file
            bbox_lines = bbox_file.readlines()
            first_line_bbox = bbox_lines[skip_bbox_lines].split(' ')
            # check if we correctly skipped the first line of the file, the one with no number,
            # and if all the elements are numeric, like every coordinate should be ;)
            if first_line_bbox[0].isnumeric() and first_line_bbox[1].isnumeric() and \
                    first_line_bbox[2].isnumeric() and first_line_bbox[3].rstrip().isnumeric():
                x, y, width, height = first_line_bbox
                # adapt the old coordinates to the new stretched dimension
                left = int(x.strip()) * percent_width
                upper = int(y.strip()) * percent_height
                right = int(int(x.strip()) + int(width)) * percent_width
                lower = int(int(y.strip()) + int(height)) * percent_height
                # crop and resize the query image
                pil_query_img = pil_resized_target_img_bbox.crop((left, upper, right, lower))
                pil_resized_query_img = pil_query_img.resize((query_img_dim, query_img_dim))
            else:
                error_string = f"Bounding box file's first line should have 4 groups of integers with whitespace " \
                               f"separator. Check {bbox_path}"
                raise Exception(error_string)

        # Mask #

        try:
            pil_mask = Image.open(mask_img_path)
            pil_resized_mask = pil_mask.resize((img_dim, img_dim))
        except FileNotFoundError:
            target_img_width, target_img_height = pil_target_img.size
            resized_target_img_width, resized_target_img_height = pil_resized_target_img.size
            percent_width = round(100 * int(resized_target_img_width) / (int(target_img_width)), 2) / 100
            percent_height = round(100 * int(resized_target_img_height) / (int(target_img_height)), 2) / 100
            with open(mask_img_path, mode='r') as f:
                bbox_lines = f.readlines()
                bboxes = []
                for y in range(skip_bbox_lines, len(bbox_lines)):
                    x, y, width, height = bbox_lines[y].rstrip().split(' ')
                    x = int(int(x.strip()) * percent_width)
                    y = int(int(y.strip()) * percent_height)
                    width = int(int(width.strip()) * percent_width)
                    height = int(int(height.strip()) * percent_height)
                    bboxes.append((x, y, width, height))
            pil_resized_mask = get_mask_from_bbox(bboxes)
            pil_resized_mask = Image.fromarray(pil_resized_mask)

        # just to test if everything works. don't look at these :)
        # if pil_resized_target_img:
        #     pil_resized_target_img.save('target.jpg')
        # if pil_resized_query_img:
        #     pil_resized_query_img.save('query.jpg')
        # if pil_resized_mask:
        #     pil_resized_mask.save('mask.jpg')

        # return the triplet (Dq, Dt, Dm) where Dq is the query image, Dt is the target image and Dm is the mask image
        return {
            "query": to_pytorch(pil_resized_query_img),
            "target": to_pytorch(pil_resized_target_img),
            "mask": to_pytorch(pil_resized_mask)
        }

    def __getitem__(self, item_index):
        # get the path of the preprocessed file, if exists
        query_file_path = f'{self.processed_img_dir}{os.path.sep}{item_index}_query.hdf5'
        target_file_path = f'{self.processed_img_dir}{os.path.sep}{item_index}_target.hdf5'
        mask_file_path = f'{self.processed_img_dir}{os.path.sep}{item_index}_mask.hdf5'

        triplet_files = [query_file_path, target_file_path, mask_file_path]
        triplet_element_type = ["query", "target", "mask"]

        return_dict = {}
        for file_path in triplet_files:
            if not os.path.exists(file_path):
                selected_img = self.imgs_path[item_index]
                return_dict = self.preprocess(
                    target_img_path=get_full_path(self.imgs_dir, selected_img[self.TARGET_IMG_PATH]),
                    bbox_path=get_full_path(self.masks_dir, selected_img[self.BBOX_PATH]),
                    query_full_img_path=get_full_path(self.imgs_dir, selected_img[self.QUERY_FULL_IMG_PATH]),
                    mask_img_path=get_full_path(self.masks_dir, selected_img[self.MASK_IMG_PATH]),
                    skip_bbox_lines=self.skip_bbox_lines)
                if self.save_to_disk:
                    store_processed_imgs(self.processed_img_dir, return_dict, item_index)
                break
        else:
            for index, file_path in enumerate(triplet_files):
                if os.path.exists(file_path):
                    return_dict[triplet_element_type[index]] = read_processed_img(file_path)
                else:
                    raise FileNotFoundError
        return return_dict


def to_pytorch(image):
    if image:
        image_np = np.asarray(image)
        # mask image has only one channel, we need to explicit it
        if len(image_np.shape) == 2:
            image_np = np.expand_dims(image_np, axis=2)
        # HWC to CHW for pytorch
        img_trans = image_np.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255
        return torch.from_numpy(img_trans).type(torch.FloatTensor)
    else:
        return None


def get_class_file_path(class_name, file_name):
    class_file = f"{class_name[class_name.rindex(os.path.sep):]}{os.path.sep}{file_name}"
    if os.path.sep not in class_file[0:2]:
        class_file = f"{os.path.sep}{class_file}"
    return class_file


def get_full_path(root, file):
    try:
        if root[root.rindex(os.path.sep) + 1:].strip() == file[1:file.index(os.path.sep, 1)].strip():
            path = f"{root[:root.rindex(os.path.sep)]}{file}"
        else:
            path = f"{root}{file}"
        return path
    except AttributeError:
        return None


def fix_input_dir(dir):
    if not dir.strip()[-1:] == os.path.sep:
        return dir.strip()
    return dir.strip()[:-1]


def store_processed_imgs(dir, images, image_index):
    for image in images:
        file_name = f'{dir}{os.path.sep}{image_index}_{image}.hdf5'
        f = h5py.File(file_name, "w")
        # TODO: You can try other algorithms like Mafisc and Bitshuffle
        f.create_dataset("init", compression="gzip", compression_opts=9, data=images[image])
        f.close()


def read_processed_img(hdf5_file):
    with h5py.File(hdf5_file, 'r') as hf:
        data = hf.get('init')
        data = np.array(data)
    return data
