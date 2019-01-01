# nnCartoonifier

A neural network which converts images into cartoons as seen below.  
![aishwarya](docs/aishwarya.png)  
![mountain](docs/mountain.png)  
Admittedly, these look less like cartoons and more like portraits from a famous artist!  

# Dataset and training
Unlike the usual scenario, the input-output pairs used for training were generated using a python program. The program takes real images as
input and outputs a cartoonified version as shown below.  
![train1](docs/train1.png)  
![train2](docs/train2.png)
The input-output pairs generated by the program were used to train the neural network. The goal was to have the neural network learn
the cartoonifying functionality of the program.  

The model does not replicate the program output perfectly, and the end result is a portrait-like rendering of the input image rather than a
cartoonification.   
![dog](docs/dog.png)  
I personally think that the model output is way cuter than the program output :smiley:
