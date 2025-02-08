import json

from kafka import KafkaConsumer


def consumer_ratings_and_retain():
    """
    Consume messages from the Kafka topic 'ratings' and retain them in a list.
    """
    consumer = KafkaConsumer(
        'user-ratings',
        bootstrap_servers='localhost:9092',
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        group_id='python-retrain-group',
        value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    )

    for msg in consumer:
        print(msg.value)


if __name__ == "__main__":
    consumer_ratings_and_retain()
