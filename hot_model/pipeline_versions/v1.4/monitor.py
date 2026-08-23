import env
import json
import logging
import paho.mqtt.client as mqtt
from simulator.client_mock import ClientMock


class Driver:

    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.lc = logging.getLogger("app-inference")

    def start(self):
        if env.ENABLE_SIMULATOR:
            client = ClientMock()
        else:
            client = mqtt.Client(protocol=mqtt.MQTTv311)
        client.on_connect = self.on_connect
        client.on_message = self.on_message
        client.connect(env.MQTT_HOST, env.MQTT_PORT, keepalive=60)
        client.loop_forever()

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.lc.info("Connected to EdgeX MQTT broker")
            client.subscribe(env.SUB_TOPIC)
        else:
            self.lc.error("MQTT connection failed with code %s", rc)

    def on_message_v1(self, client, userdata, msg):
        self.lc.info(msg.payload.decode())

    def on_message(self, client, userdata, msg):
        event = json.loads(msg.payload.decode())
        if "payload" in event and "event" in event["payload"] and "readings" in event["payload"]["event"]:
            event = event["payload"]["event"]
            if ("deviceName" not in event):
                self.lc.info(f"skip event, missing deviceName, {json.dumps(event)}")
                return
            if ("profileName" not in event):
                self.lc.info(f"skip event, missing profileName, {json.dumps(event)}")
                return
            if ("sourceName" not in event):
                self.lc.info(f"skip event, missing sourceName, {json.dumps(event)}")
                return
            if ("origin" not in event):
                self.lc.info(f"skip event, missing origin, {json.dumps(event)}")
                return
            if ("readings" not in event):
                self.lc.info(f"skip event, missing readings, {json.dumps(event)}")
                return
            # self.lc.info(f"event {event['deviceName']} {event['profileName']} {event['sourceName']} {event['origin']} {len(event['readings'])}")
            print(json.dumps(event))


driver = Driver()
driver.start()
