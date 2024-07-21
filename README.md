# ILMCNet: A deep neural network model that uses a pre-trained model to process features and employs conditional random fields to predict protein secondary structure
##### ILMCNet is a method for protein secondary structure prediction based on a single amino acid sequence, here are some instructions for its use.
### Before performing the prediction, you need to make the following preparations:
#### - Download the model file from https://huggingface.co/Rostlab/prot_t5_xl_half_uniref50-enc/blob/main/pytorch_model.bin and store it in the model\prot_t5_xl_half_ uniref50-enc folder.
#### - If you need to repeat the training and testing, you can run get_prottrans.py first to save the npz file of the dataset. Of course, you can also not save it, but the efficiency may be slower this way.
#### - The best trained model is stored in Google Drive and can be downloaded via https://drive.google.com/drive/folders/19fO5QKLdOLYPR_qco1P8bATKoqWzsn-_?usp=sharing.
### Predicting protein secondary structure
#### - Get the amino acid sequence of the target protein, or download the corresponding fatsa file.
#### - Run python run.py, open the pop-up link, and input the sequence for prediction.
### Train the model
#### - Set is_train = True and is_single = False in args.py.
#### - run.py replace with
##### `if __name__ == '__main__':`
#####   `start()`
#### - run python run.py
