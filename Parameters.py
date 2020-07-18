class Parameters:
    def __init__(self,in_arg):
        self.in_arg = in_arg
    
    def setArguments(self,in_arg):
        self.in_arg = in_arg
    
    def getArguments(self):
        return self.in_arg

    def displayParameters(self,action):
        if action == 'training':
            print("Training a CNN ")
            print("Algorithm :",self.in_arg['arch']) 
            print("With hyperparameters: ")
            print("learningrate: ",self.in_arg['learningrate'])
            print("hidden units: ",self.in_arg['hidden_units'])
            print("epoch(s): ",self.in_arg['epochs'])
            print("location of save model {}checkpoint.pth ".format(self.in_arg['save_directory']))
            print("The training is performed in {} processor".format(self.in_arg['gpu']))
        else:
            print("Predicting the Image ")
            print("imagefilepath :",self.in_arg['imagefilepath']) 
            print("top_k: ",self.in_arg['top_k'])
            print("file for flower name dictionary | ({}) ): ".format(self.in_arg['category_names']))
            print("The predicting is performed in {} processor".format(self.in_arg['gpu']))

