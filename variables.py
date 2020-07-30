"""
Training Variables:
DATASET_RGB : path to the training RGB images.
DATASET_GTI : path to the training ground truth masks.
MODEL : do you want to fine-tune a model? Show its path here.
DEBUG_FOLDER : folder used to save debug images.
LOAD_FEW_SAMPLES : load only 4 images of the dataset to test the code.
LOAD_MODEL_WEIGHTS : if True the training function loads the MODEL for fine-tuning.
WS : window size. Note: if you change this the network must be retrained.
"""
DATASET_RGB = "/home/stefano/Workspace/data/AerialImageDataset/train/images/*.tif"
DATASET_GTI = "/home/stefano/Workspace/data/AerialImageDataset/train/gt/*.tif"
MODEL = "./saved_models/MapRepair_pretrained"
DEBUG_FOLDER = "/home/stefano/Dropbox/img/"
LOAD_FEW_DATA_SAMPLES = False
LOAD_MODEL_WEIGHTS = True
WS = 448 


"""
Evaluation Variables:
PREDICTION_MODEL : the model used for evaluation.
PREDICTION_RGB : path to the evaluation RGB images.
PREDICTION_GTI : path to the noisy evaluation masks.
OUT_FOLDER : folder used to save the results. The result has the same filename of the RGB image.
BORDER : size of the border discarded from the prediction.
"""
PREDICTION_MODEL = "./saved_models/MapRepair_pretrained"
PREDICTION_RGB = "./data/rgb/*.tif"
PREDICTION_GTI = "./data/map/*.tif"
OUT_FOLDER = "./data/out/"
BORDER = 110
