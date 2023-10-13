import numpy as np
class Loss:
    def calculate(self,output,y):
        sample_losses=self.forward(output,y)
        data_loss=np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossenrophy(Loss):
    def forward(self,y_out,y_trgt): #called as y_pred(softmax Output) and y_true(Hot encoded value)
        y_out_clipped = np.clip(y_out, 1e-7 , 1- 1e-7) 


        samples=len(y_out)
        if len(y_trgt.shape)==1: # scaler array
            correct_confidences = y_out_clipped[range(samples),y_trgt]
        elif len(y_trgt.shape)==2: # 2d array 
            correct_confidences=np.sum(y_out_clipped*y_trgt,axis=1) 
        else:
            raise Exception("Please Enter a valid target array")
        negative_log_likelihoods=-np.log(correct_confidences)
        return negative_log_likelihoods