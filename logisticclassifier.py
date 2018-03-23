from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

df= pd.read_csv("california_housing_train.csv")

df = df.reindex(np.random.permutation(df.index))

#特征函数
def preprocess_features(df):

    selected_features = df[
            ["latitude",
             "longitude",
             "housing_median_age",
             "total_rooms",
             "total_bedrooms",
             "population",
             "households",
             "median_income"]]
    processed_features = selected_features.copy()

    processed_features["rooms_per_person"] = (df["total_rooms"] /df["population"])
    return processed_features





#特征列
def construct_feature_columns(input_features):
    return set([tf.feature_column.numeric_column(my_feature) for my_feature in input_features])




#目标函数
def preprocess_targets(df):
    output_targets = pd.DataFrame()
    output_targets["median_house_value_is_high"] = (df["median_house_value"] > 265000).astype(float)
    return output_targets



#分配测试数据
train_x = preprocess_features(df.head(12000))
train_y = preprocess_targets(df.head(12000))

#分配验证数据
validation_x = preprocess_features(df.tail(3000))
validation_y = preprocess_targets(df.tail(3000))


#输入函数
def my_input_fn(x, y, batch_size=1, shuffle=True, num_epochs=None):

    x = {key:np.array(value) for key,value in dict(x).items()}
    ds = Dataset.from_tensor_slices((x,y)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)

    if shuffle:
        ds = ds.shuffle(100)

    x, labels = ds.make_one_shot_iterator().get_next()
    return x, labels



#建模
def train_linear_classifier_model(
    learning_rate,
    steps,
    batch_size,
    train_x,
    train_y,
    validation_x,
    validation_y):

    periods = 10
    steps_per_period = steps / periods

    my_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    linear_classifier = tf.estimator.LinearClassifier(
            feature_columns=construct_feature_columns(train_x),
            optimizer=my_optimizer
            )

    train_input_fn = lambda: my_input_fn(train_x, train_y["median_house_value_is_high"], batch_size=batch_size)
    predict_train_input_fn = lambda: my_input_fn(train_x, train_y["median_house_value_is_high"],num_epochs=1, shuffle=False)
    predict_validation_input_fn = lambda: my_input_fn(validation_x, validation_y["median_house_value_is_high"],num_epochs=1, shuffle=False)

    print ( "Training model...")
    print ("LogLoss (on training data):")
    train_log_losses = []
    validation_log_losses = []

    for period in range (0, periods):        
        linear_classifier.train(
                input_fn=train_input_fn,
                steps=steps_per_period
                )

        training_probabilities = linear_classifier.predict(input_fn=predict_train_input_fn)
        training_probabilities = np.array([item['probabilities'] for item in training_probabilities])

        validation_probabilities = linear_classifier.predict(input_fn=predict_validation_input_fn)
        validation_probabilities = np.array([item['probabilities'] for item in validation_probabilities])

        training_log_loss = metrics.log_loss(train_y, training_probabilities)
        validation_log_loss = metrics.log_loss(validation_y, validation_probabilities)

        print ("  period %02d : %0.2f" % (period, training_log_loss))
        train_log_losses.append(training_log_loss)
        validation_log_losses.append(validation_log_loss)
    print ("Model training finished.")


    plt.ylabel("LogLoss")
    plt.xlabel("Periods")
    plt.title("LogLoss vs. Periods")
    plt.tight_layout()
    plt.plot(train_log_losses, label="training")
    plt.plot(validation_log_losses, label="validation")
    plt.legend()

    return linear_classifier

#运行函数
linear_classifier = train_linear_classifier_model(
        learning_rate=0.000005,
        steps=50,
        batch_size=20,
        train_x=train_x,
        train_y=train_y,
        validation_x=validation_x,
        validation_y=validation_y)

predict_validation_input_fn = lambda: my_input_fn(validation_x, validation_y["median_house_value_is_high"],num_epochs=1, shuffle=False)
evaluation_metrics = linear_classifier.evaluate(input_fn= predict_validation_input_fn)
print ("AUC on the validation set: %0.2f" % evaluation_metrics['auc'])
print ("Accuracy on the validation set: %0.2f" % evaluation_metrics['accuracy'])


