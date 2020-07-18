import json
import torch
from torchvision import models

class PredictModel:
    def __init__(self):
        self.model = None
        
    def loadCheckPoint(self,device, checkpoint_path):
        """
        Loads deep learning model checkpoint.
        """
        
        if torch.cuda.is_available():
            map_location = lambda storage, loc: storage.cuda()
        else:
            map_location = 'cpu'
        
        # Load the saved file
        # #checkpoint = torch.load(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location = map_location)
        
        #self.model.arch = checkpoint.get('arch')
        self.model = getattr(models, checkpoint.get('arch').replace("",''))(pretrained = True)
        self.model.classifier = checkpoint['classifier']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.class_to_idx = checkpoint['class_to_idx']
        
        # set the vgg parameters to remain unchanged
        for param in self.model.parameters():
            param.requires_grad = False
            
        # make code agnostic: uses GPU (cuda) if available, otherwise cpu
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # make it set itself to the specified device above
        self.model.to(device)
        
    def predict(self,image, device, top_k):
        # puts model in evaluation mode
        # instead of (default)training mode 
        self.model.to(device)
        self.model.eval()
        
        # Process image to have a fitting input
        image_torch = torch.from_numpy(image).type(torch.FloatTensor).unsqueeze(dim = 0)
        image_predict = image_torch.to(device)
        # or image_torch=torch.tensor(ImageProssing.processImage()).float().unsqueeze(dim =0).type(torch.FloatTensor).to(device)
        # or image_torch = torch.from_numpy(np.expand_dims(ImageProssing.processImage(), 
                                                  #axis=0)).type(torch.FloatTensor).to(device) 
        
        # Turn off gradients to speed up this part      
        with torch.no_grad():
            logps = self.model.forward(image_predict) # now to see the image output to predicted
            ps = torch.exp(logps) # Take the exponent to get the actual possibilities
            
            #Top 5 predictions and labels
            top_probs, top_class = ps.topk(top_k, dim = 1) # Returns the highest probability of correct prediction of a class given the image
            
            # Convert to classes
            probs = top_probs.cpu().numpy()[0]
            top_class = top_class.cpu().numpy()[0]
            class_to_idx = {lab: num for num, lab in self.model.class_to_idx.items()}
            classes = [class_to_idx[label] for label in top_class]
            # or top_class = np.array(top_class.detach())[0]   
            # classes = [class_to_idx[label] for label in top_class]
        
        #top_fl = [cat_to_name[label] for label in top_labels]
        
        return probs, classes

    def displayProbability(self,probs, classes,path):
        """
        Converts two lists into a dictionary to display on screen
        """
        with open(path, 'r') as f:
            cat_to_name = json.load(f)
            flower = [cat_to_name[f] for f in classes] 
        count = 1
        for title, prob in zip(flower, probs):
            print ("Prediction {}: Flower: {} Probability: {}". format(count, title.title(),prob))
            count += 1
            