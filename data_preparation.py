#!/usr/bin/env python3
import pandas as pd
import numpy as np
import json
import os
import matplotlib
import matplotlib.pyplot as plt


def generate_band_3():
    file_path = 'data/train.json'
    new_file_path = 'data/ train3.json'

    train = pd.read_json(file_path)

    new_band_3 = []
    for i in range(len(train)):

        num_band3 = np.array(train["band_1"][i]) + np.array(train["band_2"][i])
        new_band_3.append(num_band3.tolist())

    with open(file_path, 'r') as f:
     raw_data = json.load(f)

    for i in range(len(raw_data)):
        raw_data[i]["band_3"] = new_band_3[i]

    with open(new_file_path, 'w+') as f:
        f.writelines(json.dumps(raw_data))
    del train
    print("generate_band_3 - Successfully")


def generate_train_image():

    ice_file_path = 'data/ice/'
    ship_file_path = 'data/ship/'
    file_train_gen = 'data/ train3.json'

    os.makedirs(ice_file_path)
    os.makedirs(ship_file_path)

    train = pd.read_json(file_train_gen)

    for i in range(len(train['is_iceberg'])):
        print(i+1, "/", len(train['is_iceberg']))
        if str(train['is_iceberg'][i]) == "0":
            matplotlib.image.imsave(ship_file_path + 'ship_' + str(i) + '.jpg',
                                np.array(train["band_3"][i]).reshape(75, 75))
        else:
            if str(train['is_iceberg'][i]) == "1":
                matplotlib.image.imsave(ice_file_path + 'ice_' + str(i) + '.jpg',
                                    np.array(train["band_3"][i]).reshape(75, 75))
    del train

    print("generate_train_image - Successfully")


def generate_test_image():
    image_path = 'data/test/'
    json_path = 'data/test.json'

    os.makedirs(image_path)

    test = pd.read_json(json_path)

    count = 1
    for i in range(len(test)):
        print(i, "/", len(test))
        band3 = np.array(test["band_1"][i]) + np.array(test["band_2"][i])

        if count in range(1, band3.size):
            matplotlib.image.imsave(image_path + str(count) + '.jpg',
                                    band3.reshape(75, 75))
        count += 1
    del test

    print("generate_test_image - Successfully")


def main():
    """
    All functions are called sequentiall
    """
    #generate_band_3()
    generate_train_image()
    generate_test_image()

if __name__ == "__main__":
    main()