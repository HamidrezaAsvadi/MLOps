import json
import logging
from datetime import datetime, timezone

import paho.mqtt.client as mqtt

import env
import domain

try:
    from simulator.client_mock import ClientMock
except Exception:
    ClientMock = None


class MQTTAllMonitor:
    """
    Subscribe to MQTT and print everything useful that arrives.

    This script is intentionally more verbose than monitor.py:
    - prints topic
    - prints device/profile/source/origin
    - prints raw readings
    - prints mapped readings when a mapping exists in domain.MAPPINGS
    - works with both production broker and simulator mode
    """

    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.lc = logging.getLogger("mqtt-all-monitor")

        self.device_map = {
            "24205cab-757d-4179-8be3-ba83e741d8d0": "Frontcooking Grill Left",
            "cd111143-35a9-4a74-a494-345af6f22405": "Frontcooking Grill Right",
        }

    def start(self):
        if env.ENABLE_SIMULATOR:
            if ClientMock is None:
                raise RuntimeError(
                    "ENABLE_SIMULATOR=True but simulator.client_mock.ClientMock could not be imported"
                )
            client = ClientMock()
        else:
            client = mqtt.Client(protocol=mqtt.MQTTv311)

        client.on_connect = self.on_connect
        client.on_message = self.on_message
        client.connect(env.MQTT_HOST, env.MQTT_PORT, keepalive=60)
        client.loop_forever()

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.lc.info("Connected to MQTT broker")
            client.subscribe(env.SUB_TOPIC)
            self.lc.info("Subscribed to %s", env.SUB_TOPIC)
        else:
            self.lc.error("MQTT connection failed with code %s", rc)

    def on_message(self, client, userdata, msg):
        print("\n" + "=" * 100)
        print(f"topic={msg.topic}")

        try:
            payload_text = msg.payload.decode("utf-8", errors="replace")
        except Exception:
            payload_text = str(msg.payload)

        try:
            packet = json.loads(payload_text)
        except Exception:
            print("payload_type=non_json")
            print(f"payload={payload_text}")
            return

        event = self.extract_event(packet)
        if event is None:
            print("payload_type=json_but_not_expected_edgex_event")
            print(json.dumps(packet, indent=2, ensure_ascii=False))
            return

        device_id = event.get("deviceName", "NA")
        device_label = self.device_map.get(device_id, device_id)
        profile_name = event.get("profileName", "NA")
        source_name = event.get("sourceName", "NA")
        origin = event.get("origin")
        origin_text = self.format_origin(origin)
        readings = event.get("readings", [])

        print(f"device={device_label}")
        print(f"profile={profile_name}")
        print(f"source={source_name}")
        print(f"origin_ns={origin}")
        print(f"origin_time={origin_text}")
        print(f"reading_count={len(readings)}")

        if not readings:
            print("readings=[]")
            return

        print("- raw_readings -")
        for idx, reading in enumerate(readings, start=1):
            resource = reading.get("resourceName", "NA")
            value = reading.get("value", "NA")
            value_type = reading.get("valueType", "NA")
            r_profile = reading.get("profileName", profile_name)
            r_device = self.device_map.get(reading.get("deviceName", device_id), reading.get("deviceName", device_id))
            print(
                f"[{idx}] device={r_device} | profile={r_profile} | "
                f"resource={resource} | value={value} | valueType={value_type}"
            )

        mapped = self.map_readings(readings)
        if mapped:
            print("- mapped_readings -")
            for idx, reading in enumerate(mapped, start=1):
                print(
                    f"[{idx}] device={self.device_map.get(reading['deviceName'], reading['deviceName'])} | "
                    f"resource={reading['resourceName']} | value={reading['value']}"
                )
        else:
            print("- mapped_readings -")
            print("No domain mapping matched these readings.")

        latest = self.build_latest_snapshot(mapped)
        if latest:
            print("- latest_snapshot -")
            order = ["Temperature", "Setpoint", "State", "ReleState", "Defrost"]
            ordered_parts = []
            for key in order:
                if key in latest:
                    ordered_parts.append(f"{key}={latest[key]}")
            for key, value in latest.items():
                if key not in order:
                    ordered_parts.append(f"{key}={value}")
            print(f"device={device_label} | " + " | ".join(ordered_parts))

    def extract_event(self, packet):
        if isinstance(packet, dict):
            payload = packet.get("payload")
            if isinstance(payload, dict):
                event = payload.get("event")
                if isinstance(event, dict):
                    return event

            if "readings" in packet and isinstance(packet["readings"], list):
                return packet

        return None

    def map_readings(self, readings):
        mapped = []
        for reading in readings:
            try:
                profile_name = reading.get("profileName")
                resource_name = reading.get("resourceName")
                if (
                    profile_name in domain.MAPPINGS
                    and resource_name in domain.MAPPINGS[profile_name]["mappings"]
                ):
                    mapped.append(domain.map_reading(reading.copy()))
            except Exception as exc:
                self.lc.warning("Failed to map reading %s: %s", reading, exc)
        return mapped

    def build_latest_snapshot(self, mapped_readings):
        snapshot = {}
        for reading in mapped_readings:
            snapshot[reading["resourceName"]] = reading["value"]
        return snapshot

    @staticmethod
    def format_origin(origin_ns):
        try:
            if origin_ns is None:
                return "NA"
            dt = datetime.fromtimestamp(int(origin_ns) / 1_000_000_000, tz=timezone.utc)
            return dt.isoformat()
        except Exception:
            return "NA"


if __name__ == "__main__":
    MQTTAllMonitor().start()
