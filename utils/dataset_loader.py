import os
import numpy as np
from torch.utils.data import Dataset
import logging
from PIL import Image


# TODO: Ha da funzionà co TorchVision, se hai tempo
class BasicDataset(Dataset):
    def __init__(self, imgs_dir: str, masks_dir: str, mask_image_dim=256, query_dim=64, mask_suffix='.bboxes.txt'):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.processed_img_dir = str(imgs_dir[:imgs_dir.rindex(os.path.sep)]) + os.path.sep + "processed"
        self.mask_image_dim = mask_image_dim
        self.query_dim = query_dim
        self.mask_suffix = mask_suffix
        assert mask_image_dim > 1, 'The dimension of mask and image must be higher than 1'
        assert query_dim > 1, 'The dimension of query image must be higher than 1'

        # create processed image's directory, if not exists yet
        try:
            os.mkdir(self.processed_img_dir)
        except FileExistsError:
            # some previous instance generate this directory, no need to raise an exception
            pass

        # dict with files path
        # key = incremental integer
        # value = dict
        #   key = type of path (target image, query image ,mask)
        #   value = path
        self.ids = {}

        # dict with merged masks path
        # key = target img file name
        # value = merged mask's file path
        masks_dict = {}

        # self.id key
        index = 0

        # put stuff into masks_dict
        for masks_paths, _, masks_files in os.walk(self.masks_dir):
            for mask_file_name in masks_files:
                _, mask_extension = os.path.splitext(os.path.join(masks_paths, mask_file_name))
                if mask_extension == ".png" and "merged" in mask_file_name:
                    # TODO: Non salvare tutto il path ma solo "classe/file"
                    # mask_absolute_path = os.path.join(masks_path, mask_file_name)
                    masks_dict[mask_file_name[:mask_file_name.rindex(".mask")]] = os.path.join(masks_paths,
                                                                                               mask_file_name)
        # print(masks_dict)

        # put stuff into self.ids
        for target_images_paths, _, target_images_files in os.walk(self.imgs_dir):
            for target_image_name in target_images_files:
                target_image_root_path, target_image_extension = os.path.splitext(
                    os.path.join(target_images_paths, target_image_name))
                if target_image_extension == ".jpg" and "no-logo" not in target_image_root_path:
                    # TODO: Non salvare tutto il path ma solo "classe/file"
                    self.ids[index] = {"target_image_path": os.path.join(target_images_paths, target_image_name),
                                       "mask_image_path": masks_dict[target_image_name],
                                       "bbox_path": f'{masks_dict[target_image_name][:masks_dict[target_image_name].rindex(".mask")]}{mask_suffix}'}
                    index += 1
        # print(self.ids)
        # print(len(self.ids))
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    # TODO: Calcola la shape della tripletta ma non chiamarlo "shape"
    def shape(self):
        pass

    # preprocess the images. then save in file and return a list triplet [query image, target image, mask image]. how?
    # stretch the target image
    # stretch, crop and stretch again the query image
    # stretch the mask image
    def preprocess(self, index: int, files_path: dict) -> np.ndarray:

        # extract paths from files_path
        target_image_path = files_path["target_image_path"]
        mask_image_path = files_path["mask_image_path"]
        bbox_path = files_path["bbox_path"]

        # Target image

        pil_target_image = Image.open(target_image_path)
        # stretch the image
        pil_resized_target_image = pil_target_image.resize((self.mask_image_dim, self.mask_image_dim))

        # Query image

        # we will resize, crop and resize again the image but we have the coordinates of the non resized bounding box
        original_width_target_image, original_height_target_image = pil_target_image.size
        resized_width_target_image, resized_height_target_image = pil_resized_target_image.size
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
                pil_query_image = pil_resized_target_image.crop((left, upper, right, lower))
                pil_resized_query_image = pil_query_image.resize((self.query_dim, self.query_dim))
            else:
                # TODO: nel traceback compare "error_string" e poi successivamente spiega l'eccezione. Trova un modo per togliere quel "error_string"
                error_string = f'Bounding box file\'s first line should have 4 groups of integers with whitespace separator. Check {bbox_path}'
                raise Exception(error_string)

        # Mask

        pil_mask = Image.open(mask_image_path)
        pil_resized_mask = pil_mask.resize((self.mask_image_dim, self.mask_image_dim))

        # just to test if everything works. don't look at these :)
        # pil_resized_target_image.save(f'{index}_target.jpg')
        # pil_resized_query_image.save(f'{index}_query.jpg')
        # pil_resized_mask.save(f'{index}_mask.jpg')

        # get the triplet list in a torch representation
        triplet_list_in_torch_representation = np.array(
            [to_pytorch(pil_resized_query_image), to_pytorch(pil_resized_target_image), to_pytorch(pil_resized_mask)])

        # save the file so the next time you don't have to preprocess again
        # TODO: C'è un modo più efficiente per falro, guarda qua: https://stackoverflow.com/questions/9619199/best-way-to-preserve-numpy-arrays-on-disk
        np.savez(f'{self.processed_img_dir}{os.path.sep}{index}', query=triplet_list_in_torch_representation[0],
                 target=triplet_list_in_torch_representation[1], mask=triplet_list_in_torch_representation[2])

        # return the triplet (Dq, Dt, Dm) where Dq is the query image, Dt is the target image and Dm is the mask image
        return triplet_list_in_torch_representation

    def __getitem__(self, i):
        # get the path of the preprocessed file, if exists
        file_path = f'{self.processed_img_dir}{os.path.sep}{i}.npz'

        # check if preprocessed file exists. if not, he will generate it. then return the triplet
        if os.path.exists(file_path):
            data = np.load(file_path, mmap_mode='r')
            return_tuple = np.array([data['query'], data['target'], data['mask']])
        else:
            return_tuple = self.preprocess(i, self.ids[i])
        return return_tuple


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
    return img_trans
