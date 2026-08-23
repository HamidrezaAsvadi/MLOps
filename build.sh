docker build . -t app-inference
docker run --network edgex_edgex-network app-inference