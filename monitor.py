import env
import json
import logging
import paho.mqtt.client as mqtt
from simulator.client_mock import ClientMock
import domain


class Driver:

    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.lc = logging.getLogger("app-inference")

        # Map device UUIDs to readable names
        self.device_map = {
            "24205cab-757d-4179-8be3-ba83e741d8d0": "Frontcooking Grill Left",
            "cd111143-35a9-4a74-a494-345af6f22405": "Frontcooking Grill Right",
        }

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

    def on_message(self, client, userdata, msg):
        try:
            event = json.loads(msg.payload.decode())
        except Exception as e:
            self.lc.error(f"Invalid JSON payload: {e}")
            return

        if not ("payload" in event and "event" in event["payload"] and "readings" in event["payload"]["event"]):
            return

        event = event["payload"]["event"]

        required_keys = ["deviceName", "profileName", "sourceName", "origin", "readings"]
        for key in required_keys:
            if key not in event:
                self.lc.info(f"skip event, missing {key}")
                return

        device_id = event["deviceName"]
        device_label = self.device_map.get(device_id, device_id)  # fallback to UUID if not mapped
        profile_name = event["profileName"]
        source_name = event["sourceName"]
        origin = event["origin"]
        readings = event["readings"]

        # Keep only mapped readings that matter for the model
        important = []
        for reading in readings:
            if (
                reading.get("profileName") in domain.MAPPINGS and
                reading.get("resourceName") in domain.MAPPINGS[reading["profileName"]]["mappings"]
            ):
                mapped = domain.map_reading(reading.copy())
                important.append(mapped)

        # Custom display order
        order = {
            "Temperature": 0,
            "Setpoint": 1,
            "State": 2,
            "ReleState": 3,
            "Defrost": 4,
        }

        important.sort(key=lambda r: order.get(r["resourceName"], 99))

        parts = [f"device={device_label}"]
        
        for r in important:
            parts.append(f"{r['resourceName']}={r['value']}")

        print(" | ".join(parts))


driver = Driver()
driver.start()