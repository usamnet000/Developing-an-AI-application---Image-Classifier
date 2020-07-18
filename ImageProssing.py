from PIL import Image
from torchvision import transforms
import numpy as np

class ImageProssing:
    def __init__(self,image_path):
        self.pil_im = Image.open(image_path)
    
    def processImage(self):
        # TODO: Process a PIL image for use in a PyTorch model
        transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])])
        transformed_im = transform(self.pil_im)
        np_image = np.array(transformed_im)
        np_image.transpose((2, 0, 1))
        
        return np_image
    
    def processImage2():
        # anthor way
        # Current dimensions
        width, height = self.pil_im.size
        
        # resize the images where the shortest side is 256 pixels, keeping the aspect ratio
        if width < height: resize_size=[256, 256**600]
        else: resize_size=[256**600, 256]
        self.pil_im.thumbnail(size=resize_size)
        
        # image crop into 224x224 
        #crop sizes of left,top,right,bottom
        center = width/4, height/4
        left=center[0]-(244/2)
        top=center[1]-(244/2)
        right=center[0]+(244/2)
        bottom =center[1]+(244/2)
        self.pil_im = self.pil_im.crop((left, top, right, bottom))
        
        np_image = np.array(self.pil_im)/255 # imshow() rewuires binary(0,1) so divided by 255
        
        # Normalize 
        normalise_means = [0.485, 0.456, 0.406]
        normalise_std = [0.229, 0.224, 0.225]
        np_image = (np_image-normalise_means)/normalise_std  
        
        # Seting the color to the first channel
        np_image = np_image.transpose(2, 0, 1)
        
        return np_image
