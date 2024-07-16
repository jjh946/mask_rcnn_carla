# MaskRcnn for NIA

## Author

Son Kihoon

## Usage

### Data

the data must be produced from the Carla Simulator data. The dataset must have the following directory sturcture.

- 01.raw_data
- 02.label_data
- 03.ground_truth_data

and the each of the directory must be shaped as following examples.

*01.raw_data*

```
-- Car
    |-- N01S01M01
    |   |-- Design0000
    |   |   |-- Image_RGB
    |   |   |-- PCD
    |   |   `-- Radar
    |   `-- Design0001
    |       |-- Image_RGB
    |       |-- PCD
    |       `-- Radar
    |-- N04S01M10
    |   `-- Design0000
    |       |-- Image_RGB
    |       |-- PCD
    |       `-- Radar
    |-- N06S02M05
    |   `-- Design0000
    |       |-- Image_RGB
    |       |-- PCD
    |       `-- Radar
    `-- N14S04M08
        `-- Design0000
            |-- Image_RGB
            |-- PCD
            `-- Radar
```

*02.label_data*

```
-- Car
    |-- N01S01M01
    |   |-- Design0000
    |   |   `-- outputJson
    |   `-- Design0001
    |       `-- outputJson
    |-- N04S01M10
    |   `-- Design0000
    |       `-- outputJson
    |-- N06S02M05
    |   `-- Design0000
    |       `-- outputJson
    `-- N14S04M08
        `-- Design0000
            `-- outputJson
```

*03.ground_truth_data

```
-- Car
    |-- N01S01M01
    |   |-- Design0000
    |   |   |-- Cuboid
    |   |   |-- instance_segmentation
    |   |   `-- lanedetectcamera
    |   `-- Design0001
    |       |-- Cuboid
    |       |-- instance_segmentation
    |       `-- lanedetectcamera
    |-- N04S01M10
    |   `-- Design0000
    |       |-- Cuboid
    |       |-- instance_segmentation
    |       `-- lanedetectcamera
    |-- N06S02M05
    |   `-- Design0000
    |       |-- Cuboid
    |       |-- instance_segmentation
    |       `-- lanedetectcamera
    `-- N14S04M08
        `-- Design0000
            |-- Cuboid
            |-- instance_segmentation
            `-- lanedetectcamera
```

### Preprocessing

- *NIA_data_splitter.py* is the python file which splits the dataset into train, validation, test. 
- 

