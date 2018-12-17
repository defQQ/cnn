#!/usr/bin/env python3
import os
import shutil

# separate data
# train:validation -> 8:2


def split_data():
    train_ice_path = 'data/train/ice/'
    train_ship_path = 'data/train/ship/'
    val_ice_path = 'data/val/ice/'
    val_ship_path = 'data/val/ship/'

    ice_image_path = 'data/ice/'
    ship_image_path = 'data/ship/'


    os.makedirs(train_ice_path)
    os.makedirs(train_ship_path)
    os.makedirs(val_ice_path)
    os.makedirs(val_ship_path)

    files = next(os.walk(ice_image_path))[2]
    num_ice = len(files)

    print("number of ice image: ", num_ice)

    train_num = 0.8 * num_ice
    test_num = 0.2 * num_ice

    for dirpath, subdirs, files in os.walk(ice_image_path):
        count = 0
        for file in files:
            if count < train_num:
                shutil.copy2(ice_image_path + file, train_ice_path)
            else:
                shutil.copy2(ice_image_path + file, val_ice_path)

            count += 1

    files = next(os.walk(ship_image_path))[2]
    num_ship = len(files)

    print("number of ship images: ", num_ship)

    train_num = 0.8 * num_ship
    test_num = 0.2 * num_ship

    for dirpath, subdirs, files in os.walk(ship_image_path):
        count = 0
        for file in files:
            if count < train_num:
                shutil.copy2(ship_image_path + file, train_ship_path)
            else:
                shutil.copy2(ship_image_path + file, val_ship_path)
            count += 1


def main():
    split_data()


if __name__ == "__main__":
    main()