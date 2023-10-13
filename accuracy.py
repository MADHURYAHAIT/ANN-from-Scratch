import numpy as np
class accuracy:
    def __init__ (self,output_value,class_target):
        prediction=np.argmax(output_value,axis=1)
        print(prediction)
        accuracy=np.mean(prediction==class_target) 
        print("Accuracy : ", accuracy)   
    