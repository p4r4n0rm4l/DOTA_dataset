import pandas as pd
import os
import numpy as np
import CONFIG
import time
from PIL import Image


class Utils:

    def __init__(self):
        self.paths_to_images = ''
        self.objects = CONFIG.CLASSES

        self.constractor()

    def constractor(self):
        self.image_counter = 0
        self.create_folders(CONFIG.EXTRACTION_PATH, 'train')
        self.extract_needed_imgs(CONFIG.TRAIN_PATHS, 'train')
        self.make_yolo_annotations('train')
        # self.check_imgs('test_yolo/train/train.txt')

        self.image_counter = 0
        self.create_folders(CONFIG.EXTRACTION_PATH, 'validation')
        self.extract_needed_imgs(CONFIG.TRAIN_PATHS, 'validation')
        self.make_yolo_annotations('validation')
        # self.check_imgs('test_yolo/train/train.txt')

    def create_folders(self, path, folder):
        directory = os.path.join(path, folder)
        if not os.path.exists(directory):
            os.makedirs(directory)
        if not os.path.exists(os.path.join(directory, folder)):
            os.makedirs(os.path.join(directory, folder))

    def extract_needed_imgs(self, paths, what_for='train'):
        # TODO na valw e3hghsh
        # TODO na trexei parallila

        start_time = time.time()
        imgs_path = paths[0]
        labels_path = paths[1]
        print("Start extracting the usefull images....\n")

        columnnames = ['x1', 'y1', 'x2', 'y2', 'x3',
                       'y3', 'x4', 'y4', 'object', 'difficulty']
        extracted_paths = pd.DataFrame(columns=['images', 'labels'])
        for filename in os.listdir(labels_path):
            temp1 = pd.read_csv(os.path.join(labels_path, filename),
                                names=columnnames,
                                delim_whitespace=True,
                                skiprows=2)

            for obj in self.objects:
                if not (temp1[(temp1['object'] == obj)]).empty:

                    extracted_paths = extracted_paths.append({
                        'images': os.path.join(imgs_path, filename[:-3]+'png'),
                        'labels': os.path.join(labels_path, filename)
                    },  ignore_index=True)
                    break

        self.path_of_needed_imgs = os.path.join(
            CONFIG.EXTRACTION_PATH, what_for, 'needed_imgs.csv')
        extracted_paths.to_csv(self.path_of_needed_imgs, index=None)
        print("--- %s seconds ---" % (time.time() - start_time))

    def make_yolo_annotations(self, what_for='train'):
        # TODO na valw e3hghsh
        print("Start making the yolo annotations....\n")
        start_time = time.time()
        if what_for == 'train':
            img_extracted_at = os.path.join(
                CONFIG.EXTRACTION_PATH, CONFIG.DIRECTORIES[0])
        elif what_for == 'validation':
            img_extracted_at = os.path.join(
                CONFIG.EXTRACTION_PATH, CONFIG.DIRECTORIES[1])
        elif what_for == 'test':
            img_extracted_at = os.path.join(
                CONFIG.EXTRACTION_PATH, CONFIG.DIRECTORIES[2])

        img_extracted_at = img_extracted_at + '/' + what_for + '/'

        columnnames = ['x1', 'y1', 'x2', 'y2', 'x3',
                       'y3', 'x4', 'y4', 'object', 'difficulty']
        needed_imgs = pd.read_csv(self.path_of_needed_imgs, names=[
                                  'images', 'labels'], skiprows=1)
        for row in needed_imgs.itertuples():
            data_frame = pd.read_csv(
                row[2], names=columnnames, delim_whitespace=True, skiprows=2)
            temp_df = pd.DataFrame(
                columns=['object', 'xCenter', 'yCenter', 'width', 'height'])
            for data in data_frame.itertuples():
                try:
                    object_index = CONFIG.CLASSES.index(data[9])
                    x_max = max(data[1], data[3], data[5], data[7])
                    x_min = min(data[1], data[3], data[5], data[7])
                    y_max = max(data[2], data[4], data[6], data[8])
                    y_min = min(data[2], data[4], data[6], data[8])
                    xCenter = ((x_min + x_max) / 2)
                    yCenter = ((y_min + y_max) / 2)
                    width = (x_max - x_min)
                    height = (y_max - y_min)
                    temp_df = temp_df.append({'object': object_index, 'xCenter': xCenter, 'yCenter': yCenter,
                                              'width': width, 'height': height}, ignore_index=True)
                except ValueError:
                    pass

            temp_df['object'] = temp_df['object'].astype(int)
            self.split_image(temp_df, row[1], img_extracted_at)

        self.yolo_paths_to_txt(what_for)
        print("--- %s seconds ---" % (time.time() - start_time))

    def split_image(self, data_frame, path_to_img, where_to_extract):
        # TODO na valw e3hghsh

        img = Image.open(path_to_img)
        (w, h) = img.size
        arr = np.asarray(img)

        # first_iter -> height
        first_iter = h // CONFIG.INPUT_HEIGHT

        # second_iter -> width
        second_iter = w // CONFIG.INPUT_WIDTH
        for i in range(0, first_iter):
            width_min = i * CONFIG.INPUT_WIDTH
            width_max = i * CONFIG.INPUT_WIDTH + CONFIG.INPUT_WIDTH
            for j in range(0, second_iter):
                # arr[width , height]

                height_min = j * CONFIG.INPUT_HEIGHT
                height_max = j * CONFIG.INPUT_HEIGHT + CONFIG.INPUT_HEIGHT

                temp = arr[width_min:width_max, height_min:height_max]
                im = Image.fromarray(temp)
                condition = ((data_frame['xCenter'] + data_frame['height']) < height_max) & ((data_frame['xCenter']) > height_min) & \
                            ((data_frame['yCenter'] + data_frame['width']) <
                             width_max) & ((data_frame['yCenter']) > width_min)
                df2 = pd.DataFrame()
                df2 = df2.append(data_frame[condition])

                if not df2.empty:
                    df2['xCenter'] = (
                        df2['xCenter'] - height_min) / CONFIG.INPUT_HEIGHT
                    df2['yCenter'] = (
                        df2['yCenter'] - width_min) / CONFIG.INPUT_WIDTH
                    df2['height'] /= CONFIG.INPUT_HEIGHT
                    df2['width'] /= CONFIG.INPUT_WIDTH
                    file_name = str(self.image_counter)

                    df2 = self.correct_bb(df2)

                    im.save(os.path.join(where_to_extract, file_name) + '.jpg', format='jpeg',
                            quality=100, subsampling=0)
                    df2.to_csv(os.path.join(where_to_extract, file_name) + '.txt', header=None,
                               sep=' ', index=None)

                    self.paths_to_images += os.path.join(
                        'data/', where_to_extract, file_name) + '.jpg' + '\n'
                    self.image_counter += 1

    def correct_bb(self, data_frame):
        # print(data_frame)
        temp_df = pd.DataFrame(
            columns=['object', 'xCenter', 'yCenter', 'width', 'height'])
        for data in data_frame.itertuples():
            absolute_xCent = data[2] * CONFIG.INPUT_WIDTH
            absolute_yCent = data[3] * CONFIG.INPUT_HEIGHT
            absolute_width = data[4] * CONFIG.INPUT_WIDTH
            absolute_height = data[5] * CONFIG.INPUT_HEIGHT

            bbox_x_max = int((2*absolute_xCent + absolute_width)/2)
            bbox_x_min = int((2*absolute_xCent - absolute_width)/2)

            bbox_y_max = int((2*absolute_yCent + absolute_height)/2)
            bbox_y_min = int((2*absolute_yCent - absolute_height)/2)

            temp2 = True

            if bbox_x_max > CONFIG.INPUT_WIDTH:
                temp2 = False
            if bbox_y_max > CONFIG.INPUT_HEIGHT:
                temp2 = False
            if (bbox_x_min < 0) or (bbox_y_min < 0):
                temp2 = False
            if bbox_y_min < 0:
                temp2 = False

            if temp2:
                temp_df = temp_df.append({'object': data[1], 'xCenter': data[2], 'yCenter': data[3],
                                          'width': data[4], 'height': data[5]}, ignore_index=True)

        temp_df['object'] = temp_df['object'].astype(int)
        return temp_df

    def yolo_paths_to_txt(self, what_for='train'):
        # TODO na valw perigrafh
        if what_for == 'train':
            txt_path = CONFIG.TXT_PATHS[0]
        elif what_for == 'validation':
            txt_path = CONFIG.TXT_PATHS[1]
        elif what_for == 'test':
            txt_path = CONFIG.TXT_PATHS[2]

        with open(txt_path, "w") as f:
            f.write(self.paths_to_images)

    def check_imgs(self, txt_path):
        # TODO Na valw perigrafh
        count_max = 0
        count_min = 0
        columns = ['object', 'xCenter', 'yCenter', 'width', 'height']
        path = 'test_yolo/train/train/'
        for i in range(0, 8404):
            temp_path = os.path.join(path, str(i)) + '.txt'
            temp_df = pd.read_csv(temp_path, names=columns,
                                  delim_whitespace=True, index_col=False)
            for data in temp_df.itertuples():
                absolute_xCent = data[2] * CONFIG.INPUT_WIDTH
                absolute_yCent = data[3] * CONFIG.INPUT_HEIGHT
                absolute_width = data[4] * CONFIG.INPUT_WIDTH
                absolute_height = data[5] * CONFIG.INPUT_HEIGHT

                bbox_x_max = int((2*absolute_xCent + absolute_width)/2)
                bbox_x_min = int((2*absolute_xCent - absolute_width)/2)

                bbox_y_max = int((2*absolute_yCent + absolute_height)/2)
                bbox_y_min = int((2*absolute_yCent - absolute_height)/2)

                if bbox_x_max > CONFIG.INPUT_WIDTH:
                    print(f'Error xmax on {temp_path}')
                    count_max += 1
                if bbox_y_max > CONFIG.INPUT_HEIGHT:
                    print(f'Error ymax on {temp_path}')
                    count_max += 1
                if bbox_x_min < 0:
                    count_min += 1
                    print(f'Error xmin on {temp_path}')
                if bbox_y_min < 0:
                    count_min += 1
                    print(f'Error ymin on {temp_path}')
        print(count_min, count_max)


util = Utils()
