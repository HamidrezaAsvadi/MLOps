import env
import json
import time
import logging
import paho.mqtt.client as mqtt
from dataclasses import asdict, is_dataclass
from cold_model.cold_model import ColdModel
from hot_model.hot_model import HotModel
from simulator.client_mock import ClientMock
import domain


class Driver:

    def __init__(self):
        self.cache: dict = {}
        self.cold_model = ColdModel()
        self.hot_model = HotModel()
        logging.basicConfig(level=logging.WARNING)
        self.lc = logging.getLogger("app-inference")
        self.log_counter = 0

        self.hot_allowed_device_ids = {
            "24205cab-757d-4179-8be3-ba83e741d8d0",  # Frontcooking Grill Left
            "cd111143-35a9-4a74-a494-345af6f22405",  # Frontcooking Grill Right
        }

        self.last_hot_infer_ns = 0
        self.hot_infer_every_ns = 10 * env.NS  # run HOT at most once every 10 seconds

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
        # Process only device LiveValues topics
        if not msg.topic.endswith("/LiveValues"):
            return

        self.lc.info(f"MQTT message received on topic: {msg.topic}")

        try:
            event = json.loads(msg.payload.decode())
        except Exception as e:
            self.lc.error(f"Failed to decode MQTT payload: {e}")
            return

        if not ("payload" in event and "event" in event["payload"] and "readings" in event["payload"]["event"]):
            return

        hot_trigger_needed = False
        cold_triggers = set()
        latest_origin = None

        for reading in event["payload"]["event"]["readings"]:
            if (
                "origin" not in reading or
                "deviceName" not in reading or
                "resourceName" not in reading or
                "profileName" not in reading or
                "valueType" not in reading or
                "value" not in reading or
                reading["profileName"] not in domain.MAPPINGS or
                reading["resourceName"] not in domain.MAPPINGS[reading["profileName"]]["mappings"]
            ):
                continue

            reading = domain.map_reading(reading)
            self.add_to_cache(reading)

            device_id = reading["deviceName"]
            profile_name = reading["profileName"]
            model_class = domain.MAPPINGS[profile_name]["model"]

            if latest_origin is None or reading["origin"] > latest_origin:
                latest_origin = reading["origin"]

            if model_class == "HOT":
                if device_id in self.hot_allowed_device_ids:
                    hot_trigger_needed = True
            else:
                cold_triggers.add((device_id, profile_name))

        self.log_cache_status()

        if hot_trigger_needed and latest_origin is not None:
            if latest_origin - self.last_hot_infer_ns >= self.hot_infer_every_ns:
                self.on_inference("24205cab-757d-4179-8be3-ba83e741d8d0", "Evco-EVJ705_2")
                self.last_hot_infer_ns = latest_origin

        for device_id, profile_name in cold_triggers:
            self.on_inference(device_id, profile_name)

    def add_to_cache(self, reading: dict):
        device_id = reading["deviceName"]
        origin = reading["origin"]
        resource_name = reading["resourceName"]
        profile_name = reading["profileName"]
        value = reading["value"]

        if device_id not in self.cache:
            self.cache[device_id] = {
                "start": origin,
                "end": origin,
                "profile": profile_name,
                "resources": {}
            }

        if resource_name not in self.cache[device_id]["resources"]:
            self.cache[device_id]["resources"][resource_name] = {
                "start": origin,
                "end": origin,
                "readings": []
            }

        resource = self.cache[device_id]["resources"][resource_name]

        # Deduplicate exact repeated readings
        # Check only the tail to keep it cheap
        for r in reversed(resource["readings"][-20:]):
            if r["origin"] == origin and r["value"] == value:
                return

        resource["readings"].append(reading)
        resource["readings"].sort(key=lambda x: x["origin"])
        resource["end"] = max(resource["end"], origin)

        five_minutes_offset = env.RETENTION_POLICY_MIN * 60 * env.NS
        window_start = resource["end"] - five_minutes_offset

        # Keep only readings inside the 5-minute window
        kept = [r for r in resource["readings"] if r["origin"] >= window_start]

        # Add one synthetic boundary point from the latest reading before the window
        prev = None
        for r in resource["readings"]:
            if r["origin"] < window_start:
                prev = r
            else:
                break

        if prev is not None:
            synthetic = dict(prev)
            synthetic["origin"] = window_start
            kept = [synthetic] + kept

        resource["readings"] = kept
        resource["start"] = window_start if kept else origin

        # Update device-level bounds from all resources
        starts = []
        ends = []
        for res in self.cache[device_id]["resources"].values():
            if len(res["readings"]) > 0:
                starts.append(res["start"])
                ends.append(res["end"])

        if starts:
            self.cache[device_id]["start"] = min(starts)
        if ends:
            self.cache[device_id]["end"] = max(ends)

    def log_cache_status(self):
        try:
            if self.log_counter == 0:
                self.lc.info("")
                for device_id, device in self.cache.items():
                    for resource_name, resource in device["resources"].items():
                        count = len(resource["readings"])
                        delta_s = int((resource["end"] - resource["start"]) / env.NS)
                        self.lc.info(
                            f"{device_id.rjust(40)} "
                            f"{device['profile'].rjust(20)} "
                            f"{resource_name.rjust(15)} "
                            f"[delta {delta_s}s] [# {count}]"
                        )
                    self.lc.info("-" * 125)
                self.lc.info("")

            self.log_counter = (self.log_counter + 1) % env.LOGGING_RATE

        except Exception as e:
            self.lc.error(f"Error logging cache status: {e}")

    def on_inference(self, device_id: str, profile_name: str):
        model_factory = {
            "COLD": self.cold_model,
            "HOT": self.hot_model,
        }

        model_class = domain.MAPPINGS[profile_name]["model"]
        model = model_factory[model_class]

        # ---------- HOT MODEL ----------
        if model_class == "HOT":
            if device_id not in self.hot_allowed_device_ids:
                return

            for hot_id in self.hot_allowed_device_ids:
                if hot_id not in self.cache:
                    return

            resources = [v["name"] for v in domain.MAPPINGS[profile_name]["mappings"].values()]
            five_minutes_offset = env.RETENTION_POLICY_MIN * 60 * env.NS

            merged_readings = []

            for hot_id in self.hot_allowed_device_ids:
                device = self.cache[hot_id]

                if device["end"] - device["start"] < five_minutes_offset:
                    return

                for resource_name in resources:
                    if resource_name not in device["resources"]:
                        return

                    resource = device["resources"][resource_name]
                    if len(resource["readings"]) == 0:
                        return

                    merged_readings.extend(resource["readings"])

            start = time.time()
            out = model.infer(merged_readings)
            end = time.time()
            elapsed = end - start

            self.lc.info("")
            if is_dataclass(out):
                payload = asdict(out)
            elif isinstance(out, dict):
                payload = out
            elif out is None:
                payload = None
            else:
                payload = str(out)

            #print("DEBUG type(out):", type(out), "value:", out)
            self.lc.info(f"HOT combined model_output: {json.dumps(payload, default=str)} elapsed: {elapsed:.4f}s")
            self.lc.info("")
            return

        # ---------- COLD MODEL ----------
        resources = [v["name"] for v in domain.MAPPINGS[profile_name]["mappings"].values()]
        device = self.cache[device_id]
        five_minutes_offset = env.RETENTION_POLICY_MIN * 60 * env.NS

        if device["end"] - device["start"] < five_minutes_offset:
            return

        readings = []
        for resource_name in resources:
            if resource_name not in device["resources"]:
                return

            resource = device["resources"][resource_name]
            if len(resource["readings"]) == 0:
                return

            readings.extend(resource["readings"])

        start = time.time()
        out = model.infer(readings)
        end = time.time()
        elapsed = end - start

        self.lc.info("")
        if is_dataclass(out):
            payload = asdict(out)
        elif isinstance(out, dict):
            payload = out
        elif out is None:
            payload = None
        else:
            payload = str(out)

        #print("DEBUG type(out):", type(out), "value:", out)
        self.lc.info(f"{device_id} {profile_name} model_output: {json.dumps(payload, default=str)} elapsed: {elapsed:.4f}s")
        self.lc.info("")


driver = Driver()
driver.start()