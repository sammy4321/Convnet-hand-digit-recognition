import numpy as np
import cv2 
import tensorflow as tf
import Tkinter
import dill
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("/tmp/data/",one_hot=True)
class Image:
	image=np.array(np.random.rand(100,100))
	imagepath="NULL"

		
class ImageProcessor():
	grayScaledImage=np.array(np.random.rand(100,100))
	normalizedImage=np.array(np.random.rand(100,100))
	thresholdedImage=np.array(np.random.rand(100,100))
	originalImage=np.array(np.random.rand(100,100))	
	def grayScalingImage(self):
		self.grayScaledImage=cv2.cvtColor(self.originalImage,cv2.COLOR_BGR2GRAY)
	def recieveImage(self,img):
		self.originalImage=img	
	def thresholding(self):
		self.thresholdedImage=cv2.adaptiveThreshold(self.grayScaledImage,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,115,1)
	def segmentation(self):
		self.thresholdedImage = self.thresholdedImage[0:28, 0:28]		
	def normalizing(self):
		self.normalizedImage=self.thresholdedImage/255
		return self.normalizedImage


class convNetBase():
	n_classes=10
	batch_size=128
	x=tf.placeholder('float',[None,784])
	y=tf.placeholder('float')	
	keep_rate=0.8
	keep_prob=tf.placeholder(tf.float32)
	cnn_model="NULL"
	with open('cnnmodel_dill.pkl') as f:
		cnn_model=dill.load(f)
	weights = {'W_conv1':tf.Variable(cnn_model.w_conv1),
               'W_conv2':tf.Variable(cnn_model.w_conv2),
               'W_fc':tf.Variable(cnn_model.w_fc),
               'out':tf.Variable(cnn_model.w_out)}
	biases = {'b_conv1':tf.Variable(cnn_model.b_conv1),
               'b_conv2':tf.Variable(cnn_model.b_conv2),
               'b_fc':tf.Variable(cnn_model.b_fc),
               'out':tf.Variable(cnn_model.b_out)}

	output=np.random.rand(2)
	hm_epochs=100		
	def forwardPropogation(self):
		self.x = tf.reshape(self.x, shape=[-1, 28, 28, 1])

		conv1 = tf.nn.relu(self.conv2d(self.x, self.weights['W_conv1'])+ self.biases['b_conv1'])
		conv1 = self.maxpool2d(conv1)
    
		conv2 = tf.nn.relu(self.conv2d(conv1, self.weights['W_conv2']) + self.biases['b_conv2'])
		conv2 = self.maxpool2d(conv2)

		fc = tf.reshape(conv2,[-1, 7*7*64])
		fc = tf.nn.relu(tf.matmul(fc, self.weights['W_fc'])+self.biases['b_fc'])
		fc = tf.nn.dropout(fc, self.keep_rate)

		self.output = tf.matmul(fc, self.weights['out'])+self.biases['out']


	def conv2d(self,x,W):
		return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
	def maxpool2d(self,x):
		return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
	
	def backPropogation(self):
		
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output,labels=self.y))
		optimizer = tf.train.AdamOptimizer().minimize(cost)
    
		hm_epochs = 100
		with tf.Session() as sess:
        		sess.run(tf.initialize_all_variables())

        		for epoch in range(hm_epochs):
            			epoch_loss = 0
            			for _ in range(int(mnist.train.num_examples/batch_size)):
                			epoch_x, epoch_y = mnist.train.next_batch(batch_size)
					epoch_x=epoch_x.reshape([-1, 28, 28, 1])
                		_, c = sess.run([optimizer, cost], feed_dict={self.x: epoch_x, self.y: epoch_y})
                	epoch_loss += c

            	print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)
	def accuracyTest():
        	correct = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print('Accuracy:',accuracy.eval({self.x:mnist.test.images, self.y:mnist.test.labels}))
class ourNN(convNetBase):
	imageToPredict=np.random.rand(100,100)
	predictedValue=-1
	def recieveImageToPredict(self,img):
		self.imageToPredict=img
	def predict(self):
		self.forwardPropogation()
		self.predicter=tf.argmax(self.output,1)
		self.imageToPredict=self.imageToPredict.reshape([-1,28,28,1])
		with tf.Session() as sess:	
			sess.run(tf.initialize_all_variables())	
			self.predictedValue=self.predicter.eval({self.x:self.imageToPredict})[0]
		self.displayPrediction()				
	def displayPrediction(self):
		print "Predicted Value : "
		print self.predictedValue
class ImageUpload:
	_im=Image()
	_imp=ImageProcessor()
	normalizedImage=np.array(np.random.rand(100,100))
	def askImage(self):
		self._im.imagepath=raw_input("Enter Image Name : ")
		self.storePath()
	def storePath(self):
		self._im.image=cv2.imread(self._im.imagepath,0)#Reading the Image And gray scaling is done here
	def getImageReady(self):
		self._imp.recieveImage(self._im.image)
		self._imp.segmentation()
		self.normalizedImage=self._imp.normalizing()	
	def returnImage(self):
		self._im.image=self._im.image.reshape([-1,28,28,1])
		self.normalizedImage=self._im.image
		return self.normalizedImage
		
def main():
	imu=ImageUpload()
	imu.askImage()
	imu.getImageReady()
	capturedImage=imu.returnImage()
	CNN=ourNN()
	print capturedImage.shape
	CNN.recieveImageToPredict(capturedImage)
	CNN.predict()		
				
main()


