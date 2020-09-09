import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
###




class AdaptiveThreshold:
    def __init__(self,blockSize,C):
        self.filters = tf.ones((blockSize,blockSize,1,1),dtype=tf.float32)/blockSize**2
        self.C       = tf.constant(-C,dtype=tf.float32)
    def __call__(self,inputs):
        
        #x = cv2.threshold(inputs,127,255,cv2.THRESH_BINARY)
        mean = tf.nn.conv2d(inputs, self.filters, strides = [1,1,1,1], padding='SAME')
        x = tf.where(tf.squeeze(inputs) > (tf.squeeze(mean) - self.C), tf.constant([[1]],dtype=tf.float32),tf.constant([[0]],dtype=tf.float32))
        # hint: tf.nn.conv2d, tf.where
        
        return x # return the resultant image, where 1 represents above the threshold and 0 represents below the threshold
    
class Thinning:
    
    def __init__(self):   
        self.filter1,self.filter2,self.filter3,self.filter4,self.filter5, self.filters = self._surface_patterns() 

    @staticmethod
    def _surface_patterns():

        filter1 = tf.constant([[1,1,1],[1,0,1],[1,1,1]])
        filter2 = tf.constant([[0,1,0],[0,0,1],[0,1,0]])
        filter3 = tf.constant([[0,0,0],[1,0,1],[0,0,0]])
        filter4 = tf.constant([[0,1,0],[1,0,1],[0,0,0]])
        filter5 = tf.constant([[0,1,0],[1,0,0],[0,1,0]])
        
        filters = tf.constant([[[-1,1,0],[0,0,0],[0,0,0]],[[0,0,-1],[0,0,1],[0,0,0]],[[0,0,-1],[0,0,1],[0,0,0]],[[0,0,0],[0,0,-1],[0,0,1]],[[0,0,0],[0,0,0],[0,1,-1]],[[0,0,0],[0,0,0],[1,-1,0]],[[0,0,0],[0,0,0],[1,-1,0]],[[1,0,0],[-1,0,0],[0,0,0]]])
        
        return filter1, filter2, filter3, filter4, filter5, filters# for rules 1 and 2
 
    def __call__(self,input1, input2):

        #  do thinning
        #  padding is required
        x1 = tf.pad(input1,tf.constant([[0,0],[1,1],[1,1],[0,0]]),constant_values=-1.0) #image 0, 1
        x2 = tf.pad(input2,tf.constant([[0,0],[1,1],[1,1],[0,0]]),constant_values=-1.0) #image -1, 1
        while True:
            neighbor = tf.nn.conv2d(x1, self.filter1, strides = [1,1,1,1], padding='VALID')
            result1 = tf.where(2 <= tf.squeeze(neighbor) and tf.squeeze(neighbor) <= 6,tf.constant([[1]],dtype=tf.uint8),tf.constant([[0]],dtype=tf.uint8))
            
            
            count = tf.zeros(x2.shape, dtype = tf.uint8)
            for i in range(8):
                p = tf.nn.conv2d(x2, self.filters[i], strides = [1,1,1,1], padding='VALID')
                count += tf.where(tf.squeeze(p)==2,tf.constant([[1]],dtype=tf.uint8),tf.constant([[0]],dtype=tf.uint8))
            result2 = tf.where(tf.squeeze(count)==1,tf.constant([[1]],dtype=tf.uint8),tf.constant([[0]],dtype=tf.uint8))
            
            
            rule1_s1 = tf.nn.conv2d(neighbor, self.filter2, strides = [1,1,1,1], padding='VALID')
            p1 = tf.where(tf.squeeze(rule1_s1)!=3,tf.constant([[1]],dtype=tf.uint8),tf.constant([[0]],dtype=tf.uint8))
            
            rule1_s2 = tf.nn.conv2d(neighbor, self.filter3, strides = [1,1,1,1], padding='VALID')
            p2 = tf.where(tf.squeeze(rule1_s2)!=3,tf.constant([[1]],dtype=tf.uint8),tf.constant([[0]],dtype=tf.uint8))
           
            
            rule2_s1 = tf.nn.conv2d(neighbor, self.filter4, strides = [1,1,1,1], padding='VALID')
            p3 = tf.where(tf.squeeze(rule2_s1)!=3,tf.constant([[1]],dtype=tf.uint8),tf.constant([[0]],dtype=tf.uint8))
     
            
            rule2_s2 = tf.nn.conv2d(neighbor, self.filter5, strides = [1,1,1,1], padding='VALID')
            p4 = tf.where(tf.squeeze(rule2_s2)!=3,tf.constant([[1]],dtype=tf.uint8),tf.constant([[0]],dtype=tf.uint8))
     
            result3 = tf.where((p1 + p2)==2,tf.constant([[1]],dtype=tf.uint8),tf.constant([[0]],dtype=tf.uint8))
            result4 = tf.where((p3 + p4)==2,tf.constant([[1]],dtype=tf.uint8),tf.constant([[0]],dtype=tf.uint8))
            
            
            ans1 = tf.where((tf.squeeze(x2) + result1 + result2 + result3) == 4,tf.constant([[1]],dtype=tf.uint8),tf.constant([[-1]],dtype=tf.uint8))
            ans2 = tf.where((tf.squeeze(ans1) + result1 + result2 + result4) == 4,tf.constant([[1]],dtype=tf.uint8),tf.constant([[-1]],dtype=tf.uint8))
            
            e = tf.where(tf.squeeze(ans1) != tf.squeeze(ans2),tf.constant([[1]],dtype=tf.uint8),tf.constant([[0]],dtype=tf.uint8))
            
            if np.sum(np(e)) == 0:
                      break;
            
            
            # tf.nn.conv2d, tf.math.reduce_max, tf.where 
            # add your code for rule 1

            # add your code for rule 2

            # if no pixels are changed from 1 to -1, break this loop            
 
            outputs = x1[:,1:-1,1:-1,:]        
        return outputs


#下載測試影像
url      = 'https://evatronix.com/images/en/offer/printed-circuits-board/Evatronix_Printed_Circuits_Board_01_1920x1080.jpg'
testimage= tf.keras.utils.get_file('pcb.jpg',url)
    
#讀入測試影像
inputs   = cv2.imread(testimage)

#轉成灰階影像
inputs   = cv2.cvtColor(inputs,cv2.COLOR_BGR2GRAY)

#顯示測試影像
plt.figure(figsize=(20,15))
plt.imshow(inputs,cmap='gray')
plt.axis(False)
plt.show()

#轉換影像表示方式成四軸張量(sample,height,width,channel)，以便使用卷積運算。
inputs = tf.convert_to_tensor(inputs,dtype=tf.float32)
inputs = inputs[tf.newaxis,:,:,tf.newaxis]

#使用卷積運算製作AdatpiveThresholding
binary = AdaptiveThreshold(61,-8)(inputs)

#存下AdaptiveThresholding結果
outputs = tf.where(tf.squeeze(binary)>0,tf.constant([[255]],dtype=tf.uint8),tf.constant([[0]],dtype=tf.uint8))
cv2.imwrite('pcb_threshold.png',outputs.numpy())

#顯示AdaptiveThresholding結果
plt.figure(figsize=(20,15))    
plt.imshow(tf.squeeze(binary).numpy()*255,cmap='gray')
plt.axis(False)
plt.show()

#使用卷積運算製作Thinning (改成-1代表0,1代表1會比較好算)
binary2  = binary*2-1
outputs = tf.where(tf.squeeze(Thinning()(binary,binary2))>0,tf.constant([[255]],dtype=tf.uint8),tf.constant([[0]],dtype=tf.uint8))
outputs = tf.squeeze(outputs)

#存下細線化結果
cv2.imwrite('pcb_thinning.png',outputs.numpy())

#注意由於螢幕解析度，同學在螢幕上看到的細線化結果可能不是真正結果，此時必須看存下來的結果影像。
plt.figure(figsize=(20,15))        
plt.imshow(outputs.numpy(),cmap='gray')
plt.axis(False)
plt.show()