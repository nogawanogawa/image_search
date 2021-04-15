# streamlit_image_search_sample

image_search sample project

# Installation

1. downlaod data

    1. Download data from https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
    2. put images/* dir into data/

# Usage

## prepare

```bash
docker-compose up
```

Wait for standing up of elasticsearch, and then type command below.

```bash
docker-compose run etl python main.py run 
```

## server run 

```bash
docker-compose up
```

Wait for standing up of elasticsearch, and then access at http://localhost:8501

# License

"image_search" is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).

