import cv2
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
import os 

#Constants 
SOURCE_PATH = 'Parking'
FIGSIZE = (20,10)
NROWS = 5
NCOLS = 2 
TRAIN_PATH = 'Train'
TEST_PATH = 'Test'
RANDOM_STATE = 42
TEST_SIZE = 0.3
CATEGORY_DICT = {'Yes':1, 'No':0}
INV_CAT_DICT = dict(zip(CATEGORY_DICT.values(), CATEGORY_DICT.keys()))
TARGET_SIZE = (533, 260)
TOTAL_PIXELS = 533*260

def create_data(filepath = SOURCE_PATH,train_path = TRAIN_PATH, test_path = TEST_PATH, test_size= TEST_SIZE, random_state = RANDOM_STATE):
    """Crerates train and test set for a given directory with classes 
    Directory structure should be 
    ```
    -SOURCE_PATH -Category1
                 -Category 2 ..... Category n
    ```

    Args:
        filepath (str, optional): File path of folder containing the source directory from which pictures should be extracted. Defaults to SOURCE_PATH.
        train_path (str, optional): File path of folder where train set is to be saved . Defaults to TRAIN_PATH.
        test_path (str, optional): File path of folder where test set is to be saved. Defaults to TEST_PATH.
        test_size (float, optional): Proportion of test set. Defaults to TEST_SIZE.
        random_state (int, optional): Numpy seed of train test split. Defaults to RANDOM_STATE.

    Returns:
        tuple: Tuple containing train and test set
    """    
    X = []
    y = []

    categories = os.listdir(filepath)
    for category in categories:
        images = os.listdir(os.path.join(filepath, category))
        for image in images:
            img = cv2.imread(os.path.join(filepath, category, image))
            img = cv2.resize(img, TARGET_SIZE)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            X.append(img)
            y.append(CATEGORY_DICT[category])
    X = np.array(X)
    X /= 255.0
    y = np.array(y).reshape(-1,1)
    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=random_state, test_size=test_size, stratify=y)
    train_data = (X_train, y_train)
    test_data = (X_test, y_test)

    if not os.path.exists(train_path):
        os.makedirs(train_path)
    for i in range(X_train.shape[0]):
        img = X_train[i]*255
        cv2.imwrite(os.path.join(train_path, f'train_{i}.png'), img.astype(int))
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    for i in range(X_test.shape[0]):
        img = X_test[i]
        cv2.imwrite(os.path.join(test_path, f'test_{i}.png'), img.astype(int))
    return train_data, test_data

def see_random_data(array, labels,figsize=FIGSIZE,nrows=NROWS, ncols=NCOLS,seed=RANDOM_STATE):
    """Shows random images from the image array along with their labels 

    Args:
        array (np.array): The image array
        labels (np.array): The labels array
        figsize (tuple, optional): The figure size you would like to have. Defaults to FIGSIZE.
        nrows (int, optional): Number of rows in the figure. Defaults to NROWS.
        ncols (int, optional): Number of columns in the figure. Defaults to NCOLS.
        seed (int, optional): The numpy seed for shuffling. Defaults to RANDOM_STATE.
    """    
    np.random.seed(seed)
    n_images = nrows*ncols
    indices = np.random.randint(low=0, high=labels.size, size=n_images)
    random_X = array[indices].reshape(nrows, ncols, 260, 533)
    random_y = labels[indices].reshape(nrows, ncols)
    fig, ax = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols)
    for i in range(nrows):
        for j in range(ncols):
            ax[i,j].axis('off')
            ax[i,j].imshow(random_X[i,j],cmap='gray')
            ax[i,j].set_title(INV_CAT_DICT[random_y[i,j]])
    plt.tight_layout()
    plt.show()