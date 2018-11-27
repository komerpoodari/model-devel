import os
import csv
import glob
import numpy as np
import pandas as pd
import random
import cv2
from sklearn import model_selection

def create_site_csvs():
    x_train = []
    x_test = []
    x_total = []

    for a_class, a_dir in enumerate(['red', 'not_red', 'not_light']):
        for a_filename in glob.glob('./site_dataset/{}/*.jpg'.format(a_dir)):
            a_filename = '/'.join(a_filename.split('\\'))
            x_total.append([a_filename, a_class, a_dir])

    x_train, x_test = model_selection.train_test_split(x_total, test_size=0.1)

    with open('site_train.csv', 'w') as csvfile:
        a_writer = csv.writer(csvfile)
        a_writer.writerow(['path', 'class', 'color'])
        a_writer.writerows(x_train)
        print('Site Training CSV file created successfully')

    with open('site_test.csv', 'w') as csvfile:
        a_writer = csv.writer(csvfile)
        a_writer.writerow(['path', 'class', 'color'])
        a_writer.writerows(x_test)
        print('Site Testing CSV file created successfully')

    print('CSV files created successfully')

def create_sim_csvs():
    with open('sim_train.csv', 'w') as csvfile:
        a_writer = csv.writer(csvfile)
        a_writer.writerow(['path', 'class', 'color'])
        for a_class, a_dir in enumerate(['red', 'green','yellow','not_light']):
            for a_filename in glob.glob('./sim_dataset/{}/*.jpg'.format(a_dir)):
                a_filename = '/'.join(a_filename.split('\\'))
                a_writer.writerow([a_filename, a_class, a_dir])
        print('Simulator Training CSV file created successfully')

if __name__ == "__main__":
    if not os.path.exists('./site_train.csv'):
        create_site_csvs()
    else:
        print('Site Training CSV is already present')

    if not os.path.exists('./sim_train.csv'):
        create_sim_csvs()
    else:
        print('Simulator Training CSV is already present')
        