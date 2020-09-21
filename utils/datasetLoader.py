from os.path import splitext, dirname, abspath
from os import listdir
import os
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image


# TODO: Ha da funzionà co TorchVision, se hai tempo
# TODO: Prendi una query per ogni immagine e usala come query image per l'immagine. Modificata con il TODO successivo
# TODO: Prendi la prima query image, la target image e la merged_mask
class BasicDataset(Dataset):
    # TODO: Fai in modo che funzioni su più dataset. Non gli va scritto il path del singolo dataset ma deve prenderlo da solo
    def __init__(self, imgs_dir, masks_dir, mask_image_dim=128, query_dim=64, mask_suffix='.bboxes.txt'):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.processed_img_dir = str(imgs_dir[:imgs_dir.rindex(os.path.sep)]) + os.path.sep + "processed"
        self.mask_image_dim = mask_image_dim
        self.query_dim = query_dim
        self.mask_suffix = mask_suffix
        assert mask_image_dim > 1, 'The dimension of mask and image must be higher than 1'
        assert query_dim > 1, 'The dimension of query image must be higher than 1'

        # create processed directory, if not exists yet
        try:
            os.mkdir(self.processed_img_dir)
        except FileExistsError:
            # some previous instance generate this directory, no need to raise an exception
            pass

        self.ids = []
        # put in "ids" every not merged mask
        for path, _, files in os.walk(masks_dir):
            for name in files:
                if "png" in name and "merged" not in name:
                    # TODO: Non salvare tutto il path ma solo "classe/file"
                    self.ids.append(os.path.join(path, name))
        # print(len(self.ids))
        # Old version
        # self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
        #             if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    # TODO: Calcola la shape della tripletta ma non chiamarlo "shape"
    def shape(self):
        pass

    # TODO: Correggere il problema relativo alla classe HP (la cartella in "jpg" è in uppercase mentre in "masks" è in lowercase)
    @classmethod
    def preprocess(cls, index, mask, dim_img, dim_mask, processed_img_dir, mask_suffix):
        # the number in the filename is the line+1 in the boundingbox file
        number_line_bbox_file = int(mask.split(".")[-2])
        # switch between "jpg" and "masks" directory
        # TODO: Modifica sta parte che è orribile, potrebbe fare casini
        jpg_path = mask.replace("/masks/", "/jpg/")
        # cut the mask park from the filename, the new filename is the one in the "jpg" dir
        target_image = jpg_path[:jpg_path.rindex(".mask")]

        # Target image
        pil_target_image = Image.open(target_image)
        # stretch the image
        pil_resized_target_image = pil_target_image.resize((dim_img, dim_img))

        # Query image
        # we will resize, crop and resize again the image but we have the coordinates of the non resized bounding box
        original_width_target_image, original_height_target_image = pil_target_image.size
        resized_width_target_image, resized_height_target_image = pil_resized_target_image.size
        percentage_width = round(100 * int(resized_width_target_image) / (int(original_width_target_image)), 2) / 100
        percentage_height = round(100 * int(resized_height_target_image) / (int(original_height_target_image)), 2) / 100

        # open the bounding box file
        with open(f'{mask[:mask.rindex(".mask")]}{mask_suffix}') as bbox_file:
            # get the corresponding row in the bbox file
            bbox_lines = bbox_file.readlines()
            bbox_line_splitted = bbox_lines[number_line_bbox_file + 1].split(' ')
            # check if we correctly skipped the first line of the file, the one with no number
            if bbox_line_splitted[0].isnumeric():
                x, y, width, height = bbox_line_splitted
                # adapt the old coordinates to the new dimension
                left = int(x.strip()) * percentage_width
                upper = int(y.strip()) * percentage_height
                right = int(int(x.strip()) + int(width)) * percentage_width
                lower = int(int(y.strip()) + int(height)) * percentage_height
                # crop and resize the query image
                pil_query_image = pil_resized_target_image.crop((left, upper, right, lower))
                pil_resized_query_image = pil_query_image.resize((dim_mask, dim_mask))

        # Mask
        pil_mask = Image.open(mask)
        pil_resized_mask = pil_mask.resize((dim_img, dim_img))

        torch_representation = (to_pytorch(pil_resized_query_image), to_pytorch(pil_resized_target_image), to_pytorch(pil_resized_mask))
        # save the file so the next time you don't have to preprocess again
        # there is a more efficient way to to this. check on this link: https://stackoverflow.com/questions/9619199/best-way-to-preserve-numpy-arrays-on-disk
        np.savez(f'{processed_img_dir}{os.path.sep}{index}', query=torch_representation[0], target=torch_representation[1], mask=torch_representation[2])
        # return the triplet (Dq, Dt, Dm) where Dq is the query image, Dt is the target image and Dm is the mask image
        return torch_representation

    def __getitem__(self, i):
        file_path = f'{self.processed_img_dir}{os.path.sep}{i}.npz'
        if os.path.exists(file_path):
            data = np.load(file_path, mmap_mode='r')
            return_tuple = (data['query'], data['target'], data['mask'])
        else:
            return_tuple = (self.preprocess(i, self.ids[i], self.mask_image_dim, self.query_dim, self.processed_img_dir, self.mask_suffix))
        return return_tuple

    # Old version
    # @classmethod
    # def preprocess(cls, pil_img, scale):
    #     w, h = pil_img.size
    #     newW, newH = int(scale * w), int(scale * h)
    #     assert newW > 0 and newH > 0, 'Scale is too small'
    #     pil_img = pil_img.resize((newW, newH))
    #
    #     img_nd = np.array(pil_img)
    #
    #     if len(img_nd.shape) == 2:
    #         img_nd = np.expand_dims(img_nd, axis=2)
    #
    #     # HWC to CHW
    #     img_trans = img_nd.transpose((2, 0, 1))
    #     if img_trans.max() > 1:
    #         img_trans = img_trans / 255
    #
    #     return img_trans

    # Old version
    # def __getitem__(self, i):
    #     idx = self.ids[i]
    #     mask_file = glob(self.masks_dir + idx + self.mask_suffix + '.*')
    #     img_file = glob(self.imgs_dir + idx + '.*')
    #
    #     assert len(mask_file) == 1, \
    #         f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
    #     assert len(img_file) == 1, \
    #         f'Either no image or multiple images found for the ID {idx}: {img_file}'
    #     mask = Image.open(mask_file[0])
    #     img = Image.open(img_file[0])
    #
    #     assert img.size == mask.size, \
    #         f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'
    #     img = self.preprocess(img, self.scale)
    #     mask = self.preprocess(mask, self.scale)
    #
    #     return {
    #         'image': torch.from_numpy(img).type(torch.FloatTensor),
    #         'mask': torch.from_numpy(mask).type(torch.FloatTensor)
    #     }



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
    return img_trans
