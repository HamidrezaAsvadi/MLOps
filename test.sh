# only first time
docker pull python:3.10-slim
cd ./src/006_optimization_py/app_inference

# clean old container
ssh creatus@172.16.18.1
docker stop app-inference -t 0
docker container rm app-inference

# create a new fresh container
docker run -d --network edgex_edgex-network --name app-inference python:3.10-slim sleep infinity
# docker start app-inference
docker exec app-inference bash -c "python3 -m pip install numpy==1.26.4 pandas==2.3.3 scikit-learn==1.7.2 tflite-runtime==2.14.0 paho-mqtt==2.1.0 boto3==1.42.30"
docker exec app-inference mkdir app-inference
exit

# code injection
ssh creatus@172.16.18.1 "rm -rf /home/creatus/app-inference"
scp -r . creatus@172.16.18.1:/home/creatus/app-inference
ssh creatus@172.16.18.1
cd app-inference
    docker exec app-inference rm -rf /app-inference
    docker exec app-inference mkdir /app-inference
    docker cp . app-inference:/app-inference
    docker exec -it app-inference /bin/bash
    cd app-inference
    python3 main.py

python3 monitor.py