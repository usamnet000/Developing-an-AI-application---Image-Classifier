# Predict the flower class
import datetime
import argparse

from Parameters import Parameters
from ImageProssing import ImageProssing
from PredictModel import PredictModel

def command(string):
    return str (input( string ))

def greetUser():
    currentH = int( datetime.datetime.now().hour )
    if currentH >= 0 and currentH < 12:
        print( 'Good Morning!' )

    if currentH >= 12 and currentH < 18:
        print( 'Good Afternoon!' )

    if currentH >= 18 and currentH != 0:
        print( 'Good Evening!' )
        
# This function retrieves 5 Command Line Arugments from user as input from
# the user running the program from a terminal window. This function returns
# the collection of these command line arguments from the function call as
# the variable in_arg 
parser = argparse.ArgumentParser()
parser.add_argument('--imagefilepath', type = str, help = 'image path, to predict its flower class')
parser.add_argument('--checkpoint', type = str, default = 'SavedModel/checkpoint.pth', help = 'directory to load trained model | (default = SavedModel/checkpoint.pth)')

parser.add_argument('--top_k', type = int, default = 5, help = 'how many flower class predictions should be made for the picture | (default = 5)')
parser.add_argument('--category_names', type = str, default = 'cat_to_name.json' , help = 'file for flower name dictionary | (default = cat_to_name.json)')
parser.add_argument('--gpu', type = str, default = 'cuda', help = 'cuda or cpu | (default = cuda)')

arg_in = parser.parse_args()
parameters = Parameters({"imagefilepath":arg_in.imagefilepath,"checkpoint":arg_in.checkpoint,"top_k":arg_in.top_k,"category_names":arg_in.category_names,"gpu":arg_in.gpu})

# Function that greet user
greetUser()
# Function that checks command line arguments using in_arg  
parameters.displayParameters('predict')

# Ask for image filepath to make predictions
if not parameters.in_arg['imagefilepath']:
    parameters.in_arg['imagefilepath'] = command("Enter the image path, to predict its flower class \n e.g. flowers/test/101/image_07949.jpg \n Your input:")
print( 'Do you want to modify input values?' )
while True:
    print('‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡')
    print("## What you Want modify?                 ##")
    print("## 0 : modify a imagefilepath            ##")
    print("## 1 : modify a checkpoint               ##")
    print("## 2 : modify a top_k                    ##")
    print("## 3 : modify a {} location file for flower name dictionary##".format(arg_in.category_names))
    print("## 4 : modify a mode(s)                  ##")
    print("## 5 : Print Parameters                  ##")
    print("## 6 : predict                           ##")
    print("## 7 : exit                              ##")
    print('‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡‡')
    query = int(command("command: "))
    if query == 0 :
        query = command("enter image file path, the defualt ({})".format(parameters.in_arg['imagefilepath']))
        parameters.in_arg['imagefilepath'] = query
    elif query == 1 :
        query = command("enter checkpoint path, the defualt ({})".format(parameters.in_arg['checkpoint']))
        parameters.in_arg['checkpoint'] = query
    elif query == 2:
        while True:
            query = command("enter top_k, the defualt ({})".format(parameters.in_arg['top_k']))
            if query.isdigit():
                parameters.in_arg['top_k'] = query
                break
    elif query == 3 :
        query = command("enter category names path, the defualt ({})".format(parameters.in_arg['category_names']))
        parameters.in_arg['category_names'] = query
    elif query == 4:
        while True:
            query = command("enter cuda or cpu , the defualt ({})".format(parameters.in_arg['gpu']))
            if query not in ["cuda", "cpu"]:
                print("Try again! Please give a valid processor. \nValid processor: cuda, cpu")
                query = command("command: ")
            else:
                parameters.in_arg['gpu'] = query
                break
    elif query == 5:
        parameters.displayParameters('predict')
    elif query == 6:
        pm = PredictModel()
        pm.loadCheckPoint(parameters.in_arg['gpu'], parameters.in_arg['checkpoint'])
        im = ImageProssing(parameters.in_arg['imagefilepath'])
        image = im.processImage() 
        probs, classes = pm.predict(image,parameters.in_arg['gpu'],parameters.in_arg['top_k'])
        pm.displayProbability(probs, classes,parameters.in_arg['category_names'])
    elif query == 7:
        break