# Docker build
docker build -t KartRL:latest . 

docker run -it --rm -v /c/Users/vvial/GitHub/KartRL:/App -p 6006:6006 KartRL:latest

# Run image
docker run --gpu all 