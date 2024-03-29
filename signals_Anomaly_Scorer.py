#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ {
# Name             :    signals_Anomaly_Scorer
# Authour          :    K. SrePadmashiny
# Reviewer         :    Signals Professor
# Date             :    31-Oct-2023
# Purpose          :    Signals Anomoly Detection Scoring
# Data             :    ECG Data from ecg kafka producer
#                  :    
#                  :    
# Change History   :    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Library s Begins ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# import required libraries
import pickle
from kafka import KafkaConsumer

# Kafka topics
topic = 'signal'

#Loading model 
print("Loading pre-trained model")
AnamolyModel = pickle.load(open("C:\\signals\\data\\\ecg.pickle", 'rb'))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                                        * predicting the streaming kafka messages  *
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

consumer = KafkaConsumer('signal',bootstrap_servers=['localhost:9092'])
print("Starting ML predictions.")
for message in consumer:
    val = ((message.value).decode("utf-8")).split (",")
    val1 = [float(i) for i in val]
    predicted = AnamolyModel.predict([val1])
    print(predicted[0])
    print(message.value.decode("utf-8") +" => " + str(predicted[0]))
