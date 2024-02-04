# libraries imported
import numpy as np
import pandas as pd
import csv
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout

# Load the training dataset
instagram_df_train = pd.read_csv('insta_train.csv')


def find_ratio_of_name(name: str):
    digits = 0
    for i in name:
        if i.isdigit():
            digits += 1
    return format(digits / len(name), '.2f')


def find_no_of_words_in_name(name: str):
    return len(name.split(' '))


def name_is_equal_to_username(name: str, uname: str):
    if name.__eq__(uname):
        return 1
    else:
        return 0


def dataset_update(user_inputs: list):
    with open('insta_train.csv', 'a', newline='\n') as csvfile:
        # Create a csv writer object
        writer = csv.writer(csvfile)
        # Write the new data to the CSV file
        writer.writerow(user_inputs)
        df = instagram_df_train.drop_duplicates(keep='first')
        df.to_csv('insta_train.csv', index=False)
        # Close the file object
        csvfile.close()


def compute(inputs: list):
    # profile_pic = int(input("Is the profile pic is available or not (1 or 0) : "))
    #
    # username = input("Enter the username : ")
    # length_username = float(find_ratio_of_name(username))
    #
    # fullname = input("Enter the fullname : ")
    # fullname_words = find_no_of_words_in_name(fullname)
    #
    # length_fullname = float(find_ratio_of_name(fullname))
    #
    # name_equal_username = name_is_equal_to_username(fullname, username)
    #
    # description = input("Enter the Description : ")
    # description_length = len(description)
    #
    # external_url = int(input("Is the external URL is available or not (1 or 0) : "))
    #
    # private = int(input("Is the account is private or not (1 or 0) : "))
    #
    # posts = int(input("No.of posts : "))
    #
    # followers = int(input("No of Followers : "))
    #
    # follows = int(input("No of Follows : "))
    #
    # inputs = [profile_pic, length_username, fullname_words, length_fullname,
    #           name_equal_username, description_length, external_url, private, posts,
    #           followers, follows]
    x_train = instagram_df_train.drop(columns=['fake'])

    # Training and testing dataset (Outputs)
    y_train = instagram_df_train['fake']

    # Scale the data before training the model

    scaler_x = StandardScaler()
    x_train = scaler_x.fit_transform(x_train)

    y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)

    # Building and Training Deep Training Model

    model = Sequential()
    model.add(Dense(50, input_dim=11, activation='relu'))
    model.add(Dense(150, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(150, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(25, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(2, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=25, verbose=1, validation_split=0.1, use_multiprocessing=True)

    custom_input = scaler_x.transform(np.array([inputs]))
    # custom_input = scaler_x.transform(np.array([[1, 0.44, 1, 0, 0, 0, 0, 0, 3, 39, 68]]))

    prediction = model.predict(custom_input)
    pred = np.argmax(prediction)
    if ((pred >= 0.5) * 1) == 1:
        inputs.append(1)
        dataset_update(inputs)
        return "Fake Profile"
    else:
        inputs.append(0)
        dataset_update(inputs)
        return "Real Profile"
