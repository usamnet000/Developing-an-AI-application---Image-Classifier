# Predict the flower class
import argparse
import torch
from torchvision import models
import numpy as np
from PIL import Image
from torchvision import transforms
import json

# Functions to predict the flower class from an image
def load_checkpoint(device, filepath, architecture):
    
    '''
    Input : device, filepath, architecture
    Operation : Loads previously saved model for further predictions
    Output : Saved model
    '''
    if torch.cuda.is_available():
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location = 'cpu'
    
    
    checkpoint = torch.load(filepath, map_location = map_location)
    
    architecture = architecture.replace("",'')
    model = getattr(models, architecture)(pretrained = True)
    epochs = checkpoint['epochs']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = checkpoint['optimizer']
    
    # set the vgg parameters to remain unchanged
    for param in model.parameters():
        param.requires_grad = False
    
    # make code agnostic: uses GPU (cuda) if available, otherwise cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # make it set itself to the specified device above
    model.to(device)
    
    return model


def preprocess_image(image_path):
    ''' 
    Input: image_path
    Operations: preprocesses the image to use as input for the model: crops, scales and normalizes the image
    Output: np.array of image    
    '''
        # TODO: Process a PIL image for use in a PyTorch model
    pil_im = Image.open(image_path)
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])
    transformed_im = transform(pil_im)
    np_image = np.array(transformed_im)
    np_image.transpose((2, 0, 1))
    return np_image



def predict(image, device, model, top_k):
    ''' 
    Input: Image_path, model, topk=5
    Predict the class (or classes) of an image using a trained deep learning model.
    Output: Pobablilty/Class Label/Index of 5 highest predicted classes 
    '''
    
        # Process image to have a fitting input
    
    image_np = torch.from_numpy(image).type(torch.FloatTensor).unsqueeze(dim = 0)
    
    # Get probabilities for input
    
    model.to(device)
    image_predict = image_np.to(device)
                    
    with torch.no_grad():
        
        model.eval()
        
        logps = model.forward(image_predict)
        ps = torch.exp(logps)
        top_probs, top_class = ps.topk(top_k, dim = 1)
        
        probs = top_probs.cpu().numpy()[0]
        top_class = top_class.cpu().numpy()[0]
        class_to_idx = {lab: num for num, lab in model.class_to_idx.items()}
        classes = [class_to_idx[i] for i in top_class]
    
    return probs, classes

def output(image_path, dictionary, probs, classes):
    
    '''
    Input: image_path, dictionary, probs, classes
    Operation: Gives a prediction output of flower class and prediction percentage
    Output: none
    '''
    with open(dictionary, 'r') as f:
        
        cat_to_name = json.load(f)
    
        flower = [cat_to_name[f] for f in classes]
    
    probs *= 100

    count = 1
    for title, prob in zip(flower, probs):
        print ("\n Prediction {}: {} {:.2f}%". format(count, title.title(), prob))
        count += 1
        
    return
        
parser = argparse.ArgumentParser()
parser.add_argument('--imagefilepath', type = str, help = 'image path, to predict its flower class')
parser.add_argument('--checkpoint', type = str, default = 'SavedModel/checkpoint.pth', help = 'directory to load trained model | (default = SavedModel/checkpoint.pth)')
parser.add_argument('--arch', type = str, default = 'vgg19', help = 'which CNN Model should be used for pretraining, your choose between vgg13, vgg16, vgg19, densenet121, densenet161, alexnet | (default = vgg19)')

parser.add_argument('--top_k', type = int, default = 3, help = 'how many flower class predictions should be made for the picture | (default = 3)')
parser.add_argument('--category_names', type = str, default = 'cat_to_name.json' , help = 'file for flower name dictionary | (default = cat_to_name.json)')
parser.add_argument('--gpu', type = str, default = 'cuda', help = 'cuda or cpu | (default = cuda)')


args = parser.parse_args()

# Run functions from functions_predict.py

print("\n Selection: Predicting the image {}: \n Giving a prediction of the {} most likely flower classes \n Using {} file as dictionary for flower classes \n The prediction is performed in {} mode \n".format(args.imagefilepath, args.top_k, args.category_names, args.gpu))

# Ask for image filepath to make predictions
if not args.imagefilepath:
    args.imagefilepath = str(input("Enter the image path, to predict its flower class \n e.g. flowers/test/101/image_07949.jpg \n Your input:"))

model = load_checkpoint(args.gpu, args.checkpoint, args.arch)

image = preprocess_image(args.imagefilepath)

probs, classes = predict(image, args.gpu, model, args.top_k)

output(args.imagefilepath, args.category_names, probs, classes)



