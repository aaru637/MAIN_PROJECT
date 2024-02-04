# # libraries imported
# import pandas as pd
# import numpy as np
#
# import tensorflow as tf
# from tensorflow import keras
# from keras.layers import Dense, Activation, Dropout
# from keras.optimizers import Adam
# from keras.metrics import Accuracy
#
# from keras.models import model_from_json
# from sklearn import metrics
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import classification_report, accuracy_score, roc_curve, confusion_matrix
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
#
# # Load the training dataset
# instagram_df_train = pd.read_csv('insta_train.csv')
# # print(instagram_df_train)
# test = {
#     "profile pic": 1,
#     "nums/length username": 0.33,
#     "fullname words": 1,
#     "nums/length fullname": 0.33,
#     "name==username": 1,
#     "description length": 30,
#     "external URL": 0,
#     "private": 1,
#     "#posts": 35,
#     "#followers": 488,
#     "#follows": 604,
#     "fake": 0
# }
#
# # json_file = open('model.json', 'r')
# # loaded_model_json = json_file.read()
# # json_file.close()
# # loaded_model = model_from_json(loaded_model_json)
# # loaded_model.load_weights("insta_train.csv")
#
#
# # custom_input = np.array([[1, 0.21, 2, 0, 0, 66, 1, 0, 14, 65, 33]])
# custom_input = np.array([[1, 0.21, 2, 0.12, 1, 30, 0, 1, 20, 650, 400]])
# # Load the testing data
# instagram_df_test = pd.read_csv('insta_test.csv')
# # print(instagram_df_test)
# #
# # instagram_df_train.head()
# # instagram_df_train.tail()
# #
# # instagram_df_test.head()
# # instagram_df_test.tail()
#
# # Performing Exploratory Data Analysis EDA
#
# # Getting dataframe info
# # print(instagram_df_train.info())
#
# # Get the statistical summary of the dataframe
# # print(instagram_df_train.describe())
#
# # # Checking if null values exist
# # instagram_df_train.isnull().sum()
#
# # Get the number of unique values in the "profile pic" feature
# # print(instagram_df_train['profile pic'].value_counts())
#
# # Get the number of unique values in "fake" (Target column)
# # print(instagram_df_train['fake'].value_counts())
#
# # print(instagram_df_test.info())
#
# # print(instagram_df_test.describe())
#
# # print(instagram_df_test.isnull().sum())
#
# # print(instagram_df_test['fake'].value_counts())
#
# # Perform Data Visualizations
#
# '''
# # Visualize the data
# sns.countplot(instagram_df_train['fake'])
# plt.show()
#
# # Visualize the private column data
# sns.countplot(instagram_df_train['private'])
# plt.show()
#
# # Visualize the "profile pic" column data
# sns.countplot(instagram_df_train['profile pic'])
# plt.show()
#
# # Visualize the data
# plt.figure(figsize=(20, 10))
# sns.displot(instagram_df_train['nums/length username'])
# plt.show()
# '''
#
# '''
# # Correlation plot
# plt.figure(figsize=(20, 20))
# cm = instagram_df_train.corr()
# ax = plt.subplot()
# # heatmap for correlation matrix
# sns.heatmap(cm, annot=True, ax=ax)
# plt.show()
#
# sns.countplot(instagram_df_test['fake'])
#
# sns.countplot(instagram_df_test['private'])
#
# sns.countplot(instagram_df_test['profile pic'])
# '''
#
# # Preparing Data to Train the Model
#
# # Training and testing dataset (inputs)
# X_train = instagram_df_train.drop(columns=['fake'])
# X_test = instagram_df_test.drop(columns=['fake'])
# # print(X_train)
#
# # print(X_test)
#
# # Training and testing dataset (Outputs)
# y_train = instagram_df_train['fake']
# y_test = instagram_df_test['fake']
#
# # print(y_train)
#
# # print(y_test)
#
# # Scale the data before training the model
#
# scaler_x = StandardScaler()
# X_train = scaler_x.fit_transform(X_train)
# X_test = scaler_x.transform(X_test)
#
# y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)
# y_test = tf.keras.utils.to_categorical(y_test, num_classes=2)
#
# # print(y_train)
#
# # print(y_test)
#
# # print the shapes of training and testing datasets
# # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
#
# # Training_data = len(X_train) / (len(X_test) + len(X_train)) * 100
# # print(Training_data)
#
# # Testing_data = len(X_test) / (len(X_test) + len(X_train)) * 100
# # print(Testing_data)
#
# # Building and Training Deep Training Model
#
# from tensorflow import keras
# from keras.models import Sequential
# from keras.layers import Dense, Dropout
#
# model = Sequential()
# model.add(Dense(50, input_dim=11, activation='relu'))
# model.add(Dense(150, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(150, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(25, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(2, activation='softmax'))
#
# # print(model.summary())
#
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#
# epochs_hist = model.fit(X_train, y_train, epochs=50, verbose=1, validation_split=0.1, use_multiprocessing=True)
#
# # model = Sequential()
# # model.add(Dense(50, input_dim=11, activation='relu'))
# # model.add(Dense(150, activation='relu'))
# # model.add(Dropout(0.3))
# # model.add(Dense(25, activation='relu'))
# # model.add(Dropout(0.3))
# # model.add(Dense(25, activation='relu'))
# # model.add(Dropout(0.3))
# # model.add(Dense(2, activation='softmax'))
# # print(model.summary())
#
# '''
# # Access the Performance of the model
#
# print(epochs_hist.history.keys())
#
# plt.plot(epochs_hist.history['loss'])
# plt.plot(epochs_hist.history['val_loss'])
#
# plt.title('Model Loss Progression During Training/Validation')
# plt.ylabel('Training and Validation Losses')
# plt.xlabel('Epoch Number')
# plt.legend(['Training Loss', 'Validation Loss'])
# plt.show()
# '''
#
# # print(X_test)
# # custom_input_scaled = scaler_x.transform(custom_input)
# # output = model.predict(custom_input_scaled)
# # predicted = model.predict(X_test)
# # predicted_value = []
# # test = []
# # for i in predicted:
# #     predicted_value.append(np.argmax(i))
# #
# # for i in y_test:
# #     test.append(np.argmax(i))
# #
# # print(classification_report(test, predicted_value, target_names=['Fake', 'Genuine'], zero_division="warn"))
# # # print(classification_report(output, predicted_value, target_names=['Fake', 'Genuine'], zero_division="warn"))
# # plt.figure(figsize=(10, 10))
# # cm = confusion_matrix(test, predicted_value)
# # sns.heatmap(cm, annot=True)
# # plt.show()
# # print("end")
# count = 0
# fake_count = 0
# real_count = 0
# featuress = scaler_x.transform(np.array([[1, 0.44, 1, 0, 0, 0, 0, 0, 3, 39, 68]]))
# prediction = model.predict(X_test)
# print(prediction)
# predict_value = []
# test_value = []
# for predict in prediction:
#     pred = np.argmax(predict)
#     if ((pred >= 0.5) * 1) == 1:
#         result = "The Profile is Fake"
#         fake_count += 1
#     else:
#         result = "The Profile is real"
#         real_count += 1
#     count += 1
#     print(count, end=" ")
#     print(result)
#     print('Predict', predict)
#     print('Prediction', pred)
#     print('Thresholded output', (pred >= 0.5) * 1)
#     print()
#     print()
# print("fake count", fake_count)
# print("real count", real_count)