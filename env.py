NS = 1_000_000_000
RETENTION_POLICY_MIN = 20
MQTT_PORT = 1883
SUB_TOPIC = "edgex/events/#"
LOGGING_RATE = 50

# simulation
#ENABLE_SIMULATOR = True
#MQTT_HOST = "127.0.0.1"

# production
ENABLE_SIMULATOR = False
MQTT_HOST = "edgex-mqtt-broker"
