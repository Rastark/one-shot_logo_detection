import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

class ConditioningBranch(nn.Module):

    def __init__(self, 
                logos, 
                # dataset,
                model: str = "vgg16_pre"):

        super(ConditioningBranch, self).__init__() 
        self.logos = logos
        # self.dataset = dataset
        self.model_name = model

        supported_models = {
            "vgg16": models.vgg16(),
            "vgg16_pre": models.vgg16(pretrained=True)
        }
        self.model = supported_models[model]


    # WIP
    # Gets the logo vgg16 output
    def get_vgg16_pre_output(self, logo):
        if self.model_name is not ("vgg16" or "vgg16_pre"):
            raise Exception
        # Applies a series of transformations to the image in order to classify it correctly
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),       # Resizes the image to 64x64
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])     # Mandatory to use the pretrained model
        ])
        input_tensor = preprocess(logo)
        input_batch = input_tensor.unsqueeze(0)     # Create a mini-batch as expected by the model
        
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            self.model.to('cuda')

        with torch.no_grad():
            output = self.model(input_batch)
        # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
        print(output[0])
        # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
        print(torch.nn.functional.softmax(output[0], dim=0))
        
    





        



