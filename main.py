import env
import json
import time
import logging
import io
from contextlib import redirect_stdout
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

        self.device_labels = {
            "24205cab-757d-4179-8be3-ba83e741d8d0": "Frontcooking Grill Left",
            "cd111143-35a9-4a74-a494-345af6f22405": "Frontcooking Grill Right",
        }

        self.last_hot_infer_ns = 0
        self.hot_infer_every_ns = 1 * env.NS  # run HOT at most once every 10 seconds

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

    def get_device_label(self, device_id: str) -> str:
        return self.device_labels.get(device_id, device_id)

    def get_latest_resource_values(self, device_id: str, resource_names: list[str]) -> dict:
        latest = {}
        device = self.cache.get(device_id, {})
        resources = device.get("resources", {})

        for resource_name in resource_names:
            resource = resources.get(resource_name)
            if resource and resource["readings"]:
                latest[resource_name] = resource["readings"][-1]["value"]
            else:
                latest[resource_name] = "NA"

        return latest

    def print_device_status(self, device_label: str, status: str, tau: str, latest_score: str, readings: dict, timestamp: str):
        line = (
            f"device={device_label} | "
            f"status={status} | "
            f"tau-RT={latest_score} | "
            #f"tau={tau} | "
            f"Temperature={readings.get('Temperature', 'NA')} | "
            #f"Setpoint={readings.get('Setpoint', 'NA')} | "
            f"State={readings.get('State', 'NA')} | "
            f"ReleState={readings.get('ReleState', 'NA')} | "
            f"timestamp={timestamp}"
        )
        print(line)

    def parse_hot_status_output(self, captured_output: str) -> list[tuple[str, str, str, str, str]]:
        """
        Parse hot_model stdout lines like:
            [DEBUG] device=Frontcooking Grill Left | tau=25.000000 | latest_score=0.274017
            Frontcooking Grill Left:
            2026-03-16 13:57:43.668111117, idle
        Returns list of tuples:
            (device_label, timestamp, status, tau, latest_score)
        """
        parsed = []
        lines = [line.strip() for line in captured_output.splitlines() if line.strip()]

        current_debug = {}
        i = 0

        while i < len(lines):
            line = lines[i]

            if line.startswith("[DEBUG]"):
                try:
                    parts = [p.strip() for p in line.split("|")]
                    device_label = parts[0].split("device=", 1)[1].replace("[DEBUG]", "").strip()
                    tau = parts[1].split("tau=", 1)[1].strip()
                    latest_score = parts[2].split("latest_score=", 1)[1].strip()
                    current_debug[device_label] = (tau, latest_score)
                except Exception:
                    pass
                i += 1
                continue

            if line.endswith(":") and i + 1 < len(lines) and "," in lines[i + 1]:
                device_label = line[:-1].strip()
                timestamp, status = [part.strip() for part in lines[i + 1].split(",", 1)]
                tau, latest_score = current_debug.get(device_label, ("NA", "NA"))
                parsed.append((device_label, timestamp, status, tau, latest_score))
                i += 2
                continue

            i += 1

        return parsed

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
            stdout_buffer = io.StringIO()
            with redirect_stdout(stdout_buffer):
                out = model.infer(merged_readings)
            end = time.time()
            elapsed = end - start

            captured_output = stdout_buffer.getvalue()
            # print(captured_output, end="")
            parsed_rows = self.parse_hot_status_output(captured_output)

            for hot_id in self.hot_allowed_device_ids:
                device_label = self.get_device_label(hot_id)
                latest_values = self.get_latest_resource_values(hot_id, resources)

                matched = next((row for row in parsed_rows if row[0] == device_label), None)
                if matched is None:
                    timestamp = str(device["end"] / env.NS)
                    status = "unknown"
                    tau = "NA"
                    latest_score = "NA"
                else:
                    _, timestamp, status, tau, latest_score = matched

                self.print_device_status(
                    device_label=device_label,
                    status=status,
                    tau=tau,
                    latest_score=latest_score,
                    readings=latest_values,
                    timestamp=timestamp,
                )

            self.lc.info("")
            if is_dataclass(out):
                payload = asdict(out)
            elif isinstance(out, dict):
                payload = out
            elif out is None:
                payload = None
            else:
                payload = str(out)

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

        self.lc.info(f"{device_id} {profile_name} model_output: {json.dumps(payload, default=str)} elapsed: {elapsed:.4f}s")
        self.lc.info("")


if __name__ == "__main__":
    driver = Driver()
    driver.start()
