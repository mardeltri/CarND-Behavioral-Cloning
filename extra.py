import cv2,os
import numpy as np
import matplotlib.image as mpimg
from sklearn.utils import shuffle

# Load an specific image
def load_image(data_dir, image_file):
    source_path = image_file.replace('\\','/')
    filename = source_path.split('/')[-1]
    img_path = os.path.join(data_dir,'IMG',filename)
    return mpimg.imread(img_path)

# Crop image
def crop(image):
    cropped = image[60:-25,:,:]
    return cropped

# Process image (crop, resize and change color space)
def process_img(image):
    image = crop(image)
    image = cv2.resize(image, (200, 66), cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    return image

# Translate image randomly
def translate_randomly(image,steering_angle):
    trans_x = 100 * (np.random.rand() - 0.5)
    trans_y = 10 * (np.random.rand() - 0.5)
    steering_angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steering_angle

# Flip image randomly
def flip_randomly(image, steering_angle):
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    return image, steering_angle

# Choose image
def choose_image(data_dir, sample):
    center = sample[0]
    left = sample[1]
    right = sample[2]
    steering_angle = float(sample[3])
    #print('steering_angle',steering_angle,'type',type(steering_angle))
    choice = np.random.choice(3)
    if choice == 0:
        return load_image(data_dir, left), steering_angle + 0.2
    elif choice == 1:
        return load_image(data_dir, right), steering_angle - 0.2
    return load_image(data_dir, center), steering_angle 

# Augment data with flipped and translated images
def augment(data_dir, sample):
    image, steering_angle = choose_image(data_dir, sample)
    image, steering_angle = flip_randomly(image, steering_angle)
    image, steering_angle = translate_randomly(image, steering_angle)
    return image, steering_angle

# Generator creates the bach of samples
def generator(data_folder,samples, batch_size, is_training):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                if is_training and np.random.rand() < 0.6:
                    image, steering_angle = augment(data_folder, batch_sample)
                else:
                    image = load_image(data_folder, batch_sample[0])
                    steering_angle = batch_sample[3] 
                # Process image
                images.append(process_img(image))
                angles.append(steering_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)