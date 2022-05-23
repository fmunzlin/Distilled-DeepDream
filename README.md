# Distilled-DeepDream

With this repository, I want to train a model to perform a DeepDream transformation, which can be executed on a Raspberry Pi. 

For this I utilize the approach of Mordvintsev et al. (2015) on DeepDreams, to compute stylizations, which should be learned by an Auto-encoder architecture. Here, I use a pre-trained VGG16 and a corresponding decoder to perform the transformation. After finishing training the decoder, I fix both models and add a smaller encoder, which i train as proposed by Wang et al. (2020). Afterwards, I do the same thing for my smaller decoder. 

In consequence, I am able to distill the required knowledge to perform a DeepDream transformation into a smaller model, which i am able to run on a Raspberry Pi. 
