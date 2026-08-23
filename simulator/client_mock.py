import json
import time
import env

class ClientMock():

    def __init__(self):
        self.step = 10 # number of seconds
        self.installation = {
            # cold devices
            "Evco-EV3244N9EWHXX1": {
                "64fec30e-0d4a-4c49-b8ec-c998d6c4866b",
                "ef818899-5015-406f-a25a-07ea9b40ba42"
            },
            # hot devices
            "Evco-EVJ705_2": {
                "cd111143-35a9-4a74-a494-345af6f22405",
                "24205cab-757d-4179-8be3-ba83e741d8d0"
                #"12ecda02-31e2-4cb6-8b23-93ecb323ab97"
            },
        }

    def on_connect(self, client, userdata, flags, rc):
        ...

    def on_message(self, client, userdata, msg):
        ...

    def connect(self, host, port, keepalive):
        ...

    def loop_forever(self):
        #self._bootstrap()
        time.sleep(2)
        tick = 0
        while True:
            tick += 1
            for profile_name, devices in self.installation.items():
                for device_id in devices:
                    current_time = time.time_ns()
                    self._tick(profile_name, device_id, current_time, tick)
            time.sleep(self.step)

    def _bootstrap(self):
        step = self.step * env.NS
        now = time.time_ns()
        start = now - (((5 * 60)-2) * env.NS)
        tick = 0
        for current_time in range(start, now, step):
            tick += 1
            for profile_name, devices in self.installation.items():
                for device_id in devices:
                    self._tick(profile_name, device_id, current_time, tick)

    def _tick(self, profile_name, device_id, current_time, tick):
        if profile_name == "Evco-EVJ705_2":
            message = self._build_evco_evj7052_message(profile_name, device_id, current_time, tick)
        if profile_name == "Evco-EV3244N9EWHXX1":
            message = self._build_evco_ev3244_message(profile_name, device_id, current_time, tick)
        self.on_message(self, {}, message)

    def _build_evco_evj7052_message(self, profile_name, device_id, current_time, tick):
        range = 20
        delta = float(tick % range - range/2)
        setpoint = 180.0
        message = {
            "payload": {
                "event": {
                    "deviceName": device_id,
                    "profileName": profile_name,
                    "sourceName": "LiveValues",
                    "origin": current_time,
                    "readings":
                    [
                        {
                            "origin": current_time,
                            "deviceName": device_id,
                            "resourceName": "CabinetProbe",
                            "profileName": profile_name,
                            "valueType": "Float32",
                            "value": f"{setpoint + delta}"
                        },
                        {
                            "origin": current_time,
                            "deviceName": device_id,
                            "resourceName": "TemperatureSetPoint",
                            "profileName": profile_name,
                            "valueType": "Float32",
                            "value": f"{setpoint}"
                        },
                        {
                            "origin": current_time,
                            "deviceName": device_id,
                            "resourceName": "State",
                            "profileName": profile_name,
                            "valueType": "Int32",
                            "value": "1"
                        },
                        {
                            "origin": current_time,
                            "deviceName": device_id,
                            "resourceName": "Output1",
                            "profileName": profile_name,
                            "valueType": "Uint16",
                            "value": "0"
                        }
                    ]
                }
            }
        }
        return Message(message)

    def _build_evco_ev3244_message(self, profile_name, device_id, current_time, tick):
        range = 10
        delta = float(tick % range - range/2)
        setpoint = -18.0
        message = {
            "payload": {
                "event": {
                    "deviceName": device_id,
                    "profileName": profile_name,
                    "sourceName": "LiveValues",
                    "origin": current_time,
                    "readings":
                    [
                        {
                            "origin": current_time,
                            "deviceName": device_id,
                            "resourceName": "ProbeR",
                            "profileName": profile_name,
                            "valueType": "Float32",
                            "value": f"{setpoint + delta}"
                        },
                        {
                            "origin": current_time,
                            "deviceName": device_id,
                            "resourceName": "SetPointR",
                            "profileName": profile_name,
                            "valueType": "Float32",
                            "value": f"{setpoint}"
                        },
                        {
                            "origin": current_time,
                            "deviceName": device_id,
                            "resourceName": "State",
                            "profileName": profile_name,
                            "valueType": "Int32",
                            "value": "1"
                        },
                        {
                            "origin": current_time,
                            "deviceName": device_id,
                            "resourceName": "Defrost",
                            "profileName": profile_name,
                            "valueType": "Bool",
                            "value": "false"
                        }
                    ]
                }
            }
        }
        return Message(message)

class Payload:
    def __init__(self, data: dict):
        self.data = data

    def decode(self):
        return json.dumps(self.data)

class Message:
    def __init__(self, data: dict):
        self.payload = Payload(data)
