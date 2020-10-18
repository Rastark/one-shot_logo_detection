import os
import numpy as np
from torch.utils.data import Dataset
import logging
from PIL import Image

import torch

import gzip
import shutil
import h5py
import tables


# TODO: Deve preprocessare anche le immagini di test
# TODO: Ha da funzionà co TorchVision, se hai tempo
class BasicDataset(Dataset):
    TARGET_IMAGE_PATH = "target_image_path"
    MASK_IMAGE_PATH = "mask_image_path"
    BBOX_PATH = "bbox_path"
    TARGET_IMAGE_BBOX_PATH = "target_image_bbox_path"

    def __init__(self, imgs_dir: str, masks_dir: str, mask_image_dim=256, query_dim=64, mask_suffix='.bboxes.txt',
                 save_to_disk=True, save_to_disk_with_pytorch_representation=False):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.processed_img_dir = str(imgs_dir[:imgs_dir.rindex(os.path.sep) + 1]) + "processed"
        self.mask_img_dim = mask_image_dim
        self.query_dim = query_dim
        self.mask_suffix = mask_suffix
        self.save_to_disk = save_to_disk
        self.save_to_disk_with_pytorch_representation = save_to_disk_with_pytorch_representation
        assert mask_image_dim > 1, 'The dimension of mask and image must be higher than 1'
        assert query_dim > 1, 'The dimension of query image must be higher than 1'

        if not os.path.isdir(imgs_dir):
            raise Exception("Bad path for images directory")

        if not os.path.isdir(masks_dir):
            raise Exception("Bad path for masks directory")

        # create processed image's directory, if not exists yet
        try:
            os.mkdir(self.processed_img_dir)
        except FileExistsError:
            # some previous instance generate this directory, no need to raise an exception
            pass

        # list of dict with the path of the images. here you will find the path of the following images:
        #       target, mask, bbox, target's bbox
        # every dict has 4 str key and every key has a str value
        # key = type of image
        # value = path of the image
        self.images_path = []

        self.flickrlogos32_load()

    def __len__(self):
        return len(self.images_path)

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
                    # TODO: Non salvare tutto il path ma solo "classe/file"
                    # mask_absolute_path = os.path.join(masks_path, mask_file_name)
                    masks_dict[mask_file_name[:mask_file_name.rindex(".mask")]] = os.path.join(masks_paths,
                                                                                               mask_file_name)

        # dict with every image of every class
        # key = class name
        # value = dict with images type and path
        #       key = type of image (target, query, mask)
        #       value = path of the file
        image_path_element = {}

        # put stuff into image_path_element
        for target_images_paths, _, target_images_files in os.walk(self.imgs_dir):
            # TODO: Compare come classe la cartella padre "jpg", trova un modo per risolvere
            target_image_class = target_images_paths[target_images_paths.rindex(os.path.sep) + 1:]
            for target_image_name in target_images_files:
                target_image_root_path, target_image_extension = os.path.splitext(
                    os.path.join(target_images_paths, target_image_name))
                if target_image_extension == ".jpg" and "no-logo" not in target_image_root_path:
                    # TODO: Non salvare tutto il path ma solo "classe/file"
                    x = {self.TARGET_IMAGE_PATH: os.path.join(target_images_paths, target_image_name),
                         self.MASK_IMAGE_PATH: masks_dict[target_image_name],
                         self.BBOX_PATH: f'{masks_dict[target_image_name][:masks_dict[target_image_name].rindex(".mask")]}{self.mask_suffix}'}
                    try:
                        image_path_element[target_image_class].append(x)
                    except:
                        image_path_element[target_image_class] = [x]

        # fill "images_path" variable. it generate every couple (target image, query image) for the same class
        # for now, it only skips couple (target, query) of the same image
        for target_image_class in image_path_element:
            items_class = image_path_element[target_image_class]
            for outer_image in items_class:
                bbox_path = outer_image[self.BBOX_PATH]
                outer_target_image_path = outer_image[self.TARGET_IMAGE_PATH]
                for inner_image in items_class:
                    if not outer_image == inner_image:
                        target_image_path = inner_image[self.TARGET_IMAGE_PATH]
                        mask_image_path = inner_image[self.MASK_IMAGE_PATH]
                        self.images_path.append({self.TARGET_IMAGE_PATH: target_image_path,
                                                 self.MASK_IMAGE_PATH: mask_image_path,
                                                 self.BBOX_PATH: bbox_path,
                                                 self.TARGET_IMAGE_BBOX_PATH: outer_target_image_path})
        print(len(self.images_path))

    # TODO: Calcola la shape della tripletta ma non chiamarlo "shape"
    def shape(self):
        pass

    # preprocess the images. then save in file and return a list triplet [query image, target image, mask image]. how?
    # stretch the target image
    # stretch, crop and stretch again the query image
    # stretch the mask image
    # TODO: Deve tornare una lista o un ndarray?
    def preprocess(self, index: int, files_path: dict) -> object:
        # extract paths from files_path
        target_image_path = files_path[self.TARGET_IMAGE_PATH]
        mask_image_path = files_path[self.MASK_IMAGE_PATH]
        bbox_path = files_path[self.BBOX_PATH]
        target_image_bbox_path = files_path[self.TARGET_IMAGE_BBOX_PATH]

        # Target image

        pil_target_image = Image.open(target_image_path)
        # stretch the image
        pil_resized_target_image = pil_target_image.resize((self.mask_img_dim, self.mask_img_dim))

        # Query image

        pil_target_image_bbox = Image.open(target_image_bbox_path)
        pil_resized_target_image_bbox = pil_target_image_bbox.resize((self.mask_img_dim, self.mask_img_dim))

        # we will resize, crop and resize again the image but we have the coordinates of the non resized bounding box
        original_width_target_image, original_height_target_image = pil_target_image_bbox.size
        resized_width_target_image, resized_height_target_image = pil_resized_target_image_bbox.size
        percentage_width = round(100 * int(resized_width_target_image) / (int(original_width_target_image)), 2) / 100
        percentage_height = round(100 * int(resized_height_target_image) / (int(original_height_target_image)), 2) / 100

        # open the bounding box file
        with open(bbox_path) as bbox_file:
            # read only the first line of the bbox file
            bbox_lines = bbox_file.readlines()
            first_line_bbox_splitted = bbox_lines[1].split(' ')
            # check if we correctly skipped the first line of the file, the one with no number,
            # and if all the elements are numeric, like every coordinate should be ;)
            if first_line_bbox_splitted[0].isnumeric() and first_line_bbox_splitted[1].isnumeric() and \
                    first_line_bbox_splitted[2].isnumeric() and first_line_bbox_splitted[3].rstrip().isnumeric():
                x, y, width, height = first_line_bbox_splitted
                # adapt the old coordinates to the new stretched dimension
                left = int(x.strip()) * percentage_width
                upper = int(y.strip()) * percentage_height
                right = int(int(x.strip()) + int(width)) * percentage_width
                lower = int(int(y.strip()) + int(height)) * percentage_height
                # crop and resize the query image
                pil_query_image = pil_resized_target_image_bbox.crop((left, upper, right, lower))
                pil_resized_query_image = pil_query_image.resize((self.query_dim, self.query_dim))
            else:
                # TODO: nel traceback compare "error_string" e poi successivamente spiega l'eccezione. Trova un modo per togliere quel "error_string"
                error_string = f'Bounding box file\'s first line should have 4 groups of integers with whitespace separator. Check {bbox_path}'
                raise Exception(error_string)

        # Mask

        pil_mask = Image.open(mask_image_path)
        pil_resized_mask = pil_mask.resize((self.mask_img_dim, self.mask_img_dim))

        # just to test if everything works. don't look at these :)
        # pil_resized_target_image.save('target.jpg')
        # pil_resized_query_image.save('query.jpg')
        # pil_resized_mask.save('mask.jpg')

        # now you will find a lot of stuff, we are trying to figure out which combination is better
        # seems like the best one is the one with h5py and "to_torch" after the decompression

        return_list = self.create_triplet_with_torch_representation(pil_resized_query_image,
                                                                    pil_resized_target_image,
                                                                    pil_resized_mask)
        if self.save_to_disk_with_pytorch_representation:
            triplet_list = return_list
        else:
            # get the triplet list in a simple array representation
            triplet_list = self.create_triplet_without_torch_representation(pil_resized_query_image,
                                                                            pil_resized_target_image,
                                                                            pil_resized_mask)
        if self.save_to_disk:
            # input_file = self.np_save_compressed(index, triplet_list)

            self.store_hdf5_file_with_compression(triplet_list, index)

            # self.store_hdf5_file_with_compression(triplet_list[2], index, "mask")
            # self.store_hdf5_file_with_compression(triplet_list[0], index, "query")
            # self.store_hdf5_file_with_compression(triplet_list[1], index, "target")

            # if self.save_to_disk_with_pytorch_representation:
            #     mask_file = self.h5py_with_pytorch(triplet_list[2], index, "mask")
            #     query_file = self.h5py_with_pytorch(triplet_list[0], index, "query")
            #     target_file = self.h5py_with_pytorch(triplet_list[1], index, "target")
            # else:
            #     mask_file = self.h5py_without_pytorch(triplet_list[2], index, "mask")
            #     query_file = self.h5py_without_pytorch(triplet_list[0], index, "query")
            #     target_file = self.h5py_without_pytorch(triplet_list[1], index, "target")

            # gzip_mask = self.gzip_compress(index, mask_file)
            # gzip_query = self.gzip_compress(index, query_file)
            # gzip_target = self.gzip_compress(index, target_file)
            #
            # unzipped_mask = self.gzip_uncompress(gzip_mask)
            # unzipped_query = self.gzip_uncompress(gzip_query)
            # unzipped_target = self.gzip_uncompress(gzip_target)

            # print(unzipped_target)

            # h5py_file_mask = self.read_h5py(unzipped_mask)
            # h5py_file_query = self.read_h5py(unzipped_query)
            # h5py_file_target = self.read_h5py(unzipped_target)

            # if not self.save_to_disk_with_pytorch_representation:
            #     img_mask = Image.fromarray(h5py_file_mask)
            #     img_mask.save('mask.jpg')
            #     img_query = Image.fromarray(h5py_file_query)
            #     img_query.save('query.jpg')
            #     img_target = Image.fromarray(h5py_file_target)
            #     img_target.save('target.jpg')

            # self.gzip_compress(index, input_file)

        # return the triplet (Dq, Dt, Dm) where Dq is the query image, Dt is the target image and Dm is the mask image
        return return_list

    # dude, the name says all. just read it :/
    def create_triplet_without_torch_representation(self, pil_query, pil_target, pil_mask):
        # return [np.array(pil_query), np.array(pil_target), np.array(pil_mask)]
        # return np.array([np.array(pil_query), np.array(pil_target), np.array(pil_mask)])
        return {
            "query": np.array(pil_query),
            "target": np.array(pil_target),
            "mask": np.array(pil_mask)
        }

    def create_triplet_with_torch_representation(self, pil_query, pil_target, pil_mask):
        # return [to_pytorch(pil_query), to_pytorch(pil_target), to_pytorch(pil_mask)]
        # return np.array([to_pytorch(pil_query), to_pytorch(pil_target), to_pytorch(pil_mask)])
        return {
            "query": to_pytorch(pil_query),
            "target": to_pytorch(pil_target),
            "mask": to_pytorch(pil_mask)
        }

    # def h5py_with_pytorch(self, pil_img, index, type):
    #     x = self.h5py_compression(to_pytorch(pil_img), index, type)
    #     return x
    #
    # def h5py_without_pytorch(self, pil_img, index, type):
    #     x = self.h5py_compression(pil_img, index, type)
    #     return x

    # def store_hdf5_file_with_compression(self, image, image_index, image_type):
    #     file_name = f'{self.processed_img_dir}{os.path.sep}{image_index}_{image_type}.hdf5'
    #     f = h5py.File(file_name, "w")
    #     # TODO: Esistono altri algoritmi di compressione come Mafisc. Una roba figa che puoi usare è Bitshuffle
    #     f.create_dataset("init", compression="gzip", compression_opts=9, data=image)
    #     f.close()
    #     return file_name

    def store_hdf5_file_with_compression(self, images, image_index):
        image_type = ["query", "target", "mask"]
        image_type_index = 0
        for image in images:
            file_name = f'{self.processed_img_dir}{os.path.sep}{image_index}_{image}.hdf5'
            f = h5py.File(file_name, "w")
            # TODO: Esistono altri algoritmi di compressione come Mafisc. Una roba figa che puoi usare è Bitshuffle
            f.create_dataset("init", compression="gzip", compression_opts=9, data=images[image])
            f.close()
            image_type_index += 1
        # for image in images:
        #     file_name = f'{self.processed_img_dir}{os.path.sep}{image_index}_{image_type[image_type_index]}.hdf5'
        #     f = h5py.File(file_name, "w")
        #     # TODO: Esistono altri algoritmi di compressione come Mafisc. Una roba figa che puoi usare è Bitshuffle
        #     f.create_dataset("init", compression="gzip", compression_opts=9, data=image)
        #     f.close()
        #     image_type_index += 1
        # return file_name

    # def gzip_compress(self, index, input_file):
    #     # input_file = f'{self.processed_img_dir}{os.path.sep}{index}.npz'
    #     with open(input_file, 'rb') as f_in:
    #         output_file = f'{input_file}.gz'
    #         with gzip.open(output_file, 'wb', compresslevel=9) as f_out:
    #             shutil.copyfileobj(f_in, f_out)
    #     # if os.path.exists(input_file):
    #     #     os.remove(input_file)
    #     return output_file

    # def gzip_compress(self, index, input_file):
    #     # input_file = f'{self.processed_img_dir}{os.path.sep}{index}.npz'
    #     output_file = f'{input_file}.gz'
    #     with gzip.open(output_file, 'wb', compresslevel=1) as f_out:
    #         with open(input_file, 'rb') as f_in:
    #             shutil.copyfileobj(f_in, f_out)
    #     # if os.path.exists(input_file):
    #     #     os.remove(input_file)
    #     return output_file

    # def gzip_uncompress(self, input_file):
    #     with gzip.open(input_file, 'rb') as f:
    #         file_content = f.read()
    #     output_file = input_file[:input_file.rindex('.')]
    #     with open(output_file, mode='wb') as fp:
    #         fp.write(file_content)
    #     return output_file

    def read_hdf5_file(self, hdf5_file):
        with h5py.File(hdf5_file, 'r') as hf:
            data = hf.get('init')
            data = np.array(data)
        return data

    # def np_save_compressed(self, index, triplet_list_in_torch_representation):
    #     file_name = f'{self.processed_img_dir}{os.path.sep}{index}'
    #     # save the file so the next time you don't have to preprocess again
    #     np.savez_compressed(file_name,
    #                         query=triplet_list_in_torch_representation[0],
    #                         target=triplet_list_in_torch_representation[1],
    #                         mask=triplet_list_in_torch_representation[2])
    #     return f'{file_name}.npz'

    def __getitem__(self, item_index):
        # get the path of the preprocessed file, if exists
        mask_file_path = f'{self.processed_img_dir}{os.path.sep}{item_index}_mask.hdf5'
        query_file_path = f'{self.processed_img_dir}{os.path.sep}{item_index}_query.hdf5'
        target_file_path = f'{self.processed_img_dir}{os.path.sep}{item_index}_target.hdf5'

        correct_order_triplet = [query_file_path, target_file_path, mask_file_path]
        triplet_element_order = ["query", "target", "mask"]

        return_dict = {}
        for file in correct_order_triplet:
            if not os.path.exists(file):
                return_dict = self.preprocess(item_index, self.images_path[item_index])
                break
        else:
            triplet_index = 0
            for file in correct_order_triplet:
                if os.path.exists(file):
                    hdf5_file = self.read_hdf5_file(file)
                    return_dict[triplet_element_order[triplet_index]] = to_pytorch(hdf5_file)
                triplet_index += 1
        return return_dict

        # for file in correct_order_triplet:
        # # check if preprocessed file exists. if not, he will generate it. then return the triplet
        #     if os.path.exists(file):
        #         data = np.load(file, mmap_mode='r')
        #         return_tuple = np.array([data['query'], data['target'], data['mask']])
        #     else:
        #         return_tuple = self.preprocess(i, self.images_path[i])
        # return return_tuple


# something that will be deleted
class CarvanaBasicDataset(BasicDataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        super().__init__(imgs_dir, masks_dir, scale, mask_suffix='_mask')


def to_pytorch(image):
    image_np = np.array(image)
    # mask image has only one channel, we need to explicit it
    if len(image_np.shape) == 2:
        image_np = np.expand_dims(image_np, axis=2)
    # HWC to CHW for pytorch
    img_trans = image_np.transpose((2, 0, 1))
    if img_trans.max() > 1:
        img_trans = img_trans / 255
    return torch.from_numpy(img_trans).type(torch.FloatTensor)
    # return img_trans
