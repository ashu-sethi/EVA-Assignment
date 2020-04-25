
Naming Convention of Files : Coronabatch ( Group Name ) + A5 ( Assigment 5 )  + C1...5 ( Code 1 to Code 5 where Code 5 is last step


CoronaBatch-A5C1.ipynb

# Target : 
# 99.4% (this must be consistently shown in your last few epochs, and not a one-time achievement)
# Less than or equal to 15 Epochs
# Less than 10000 Parameters

# We start by adapating our previous submission which had the following parameters 
# Batchnormalization , Dropout , 15,706 paramters and 20 epochs and it hits 99.4 % once in a while 
# Integrate with vizualizations and charts to diagnose the problem of this neural network


# Result :
# Parameters: 15.7k
# Best Train Accuracy: 99.0 (epoch 13)
# Best Test Accuracy: 99.27 (epoch 14)


# Analysis : The gain made by the network in the first epoch is around 98.2% which should be higher in order to hit 99.4% within 15 epochs.
#            So we need to reduce the number of parameters to meet the new target requirement. 
#            Training accuracy never hits 100% . Which means that we need to make some changes in the architechture .


CoronaBatch-A5C2.ipynb


# Target : 
# 99.4% (this must be consistently shown in your last few epochs, and not a one-time achievement)
# Less than or equal to 15 Epochs
# Less than 10000 Parameters

# Introduce GAP into the model and remove ending kernals to ensure that number of parameters come within the limit 

# Result :
# Parameters: 9,822
# Best Train Accuracy: 96.68 (epoch 14)
# Best Test Accuracy: 99.35 (epoch 13)


# Analysis : GAP was introduced and this removed the need to have last few convolution layers reducing parameters within target limit. 
#             
#            Training accuracy never hits 100% . Which means that we need to make some changes in the architechture .
#            We need to reorganize the architechture to see if varying the intital layers improves accuracy


CoronaBatch-A5C3.ipynb

# Target : 
# 99.4% (this must be consistently shown in your last few epochs, and not a one-time achievement)
# Less than or equal to 15 Epochs
# Less than 10000 Parameters

# Reorganize the network to increase depth after GAP so as to increase accuracy . Dropout settings need to be optimized.

# Result :
# Parameters: 12,754
# Best Train Accuracy: 98.53 (epoch 14)
# Best Test Accuracy: 99.43 (epoch 12)


# Analysis : Drop out settings were made consistent . This improved the training accuracy of the network 
#            Paramters have increased above the limitation of 10k parameters so this needs to be reduced as well


CoronaBatch-A5C4.ipynb

# Target : 
# 99.4% (this must be consistently shown in your last few epochs, and not a one-time achievement)
# Less than or equal to 15 Epochs
# Less than 10000 Parameters

# Reduce the number of parameters by changing the architecture while maintaining 99.4% accuracy using 1*1 convolution / removing kernals
# Use image augmentation techniques for better regularization

# Result :
# Parameters: 9,994 Parameters
# Best Train Accuracy: 98.08 (epoch 13)
# Best Test Accuracy: 99.43 (epoch 11)


# Analysis : Drop out settings were made consistent . We also introduced 1*1 convolution layer to manage with higher kernals while keeping number of parameters low
#            We also added bias = False into the architecture to save on some parameters . Batch size was reduced to 64 instead of 128
#            Paramters have increased above the limitation of 10k parameters so this needs to be reduced as well.
#            Image Augmentation - Image Rotation of  [ -6.1 , 6.1 ] used 

CoronaBatch-A5C5.ipynb

# Target : 
# 99.4% (this must be consistently shown in your last few epochs, and not a one-time achievement)
# Less than or equal to 15 Epochs
# Less than 10000 Parameters

# Use LR Scheduler to achieve consistent accuracy above 99.4

# Result :
# Parameters: 9,994 Parameters
# Best Train Accuracy: 98.52 (epoch 13)
# Best Test Accuracy: 99.47 (epoch 11)


# Analysis : Changed Batchsize back to 128 with no loss of accuracy 
#            Used LR of 0.05 with gamma = 0.1 and step size of 6 
#            Acheived 99.4 accuracy conssitently from Epoch 7 . 
