# MOTS
![MOTS_PIC](https://motchallenge.net/sequenceVideos/MOTS20-11-gt.jpg)

## Requirements
* Python 3.6.9
* install [requirements.txt](requirements.txt)

## Usage
1) Run 
```
python MOTS/evalMOTS.py
```



## Evaluation
To run the evaluation for your method please adjust the file ```MOTS/evalMOTS.py``` using the following arguments:

```benchmark_name```: Name of the benchmark, e.g. MOTS  
```gt_dir```: Directory containing ground truth files in ```<gt_dir>/<sequence>/gt/gt.txt```    
```res_dir```: The folder containing the tracking results. Each one should be saved in a separate .txt file with the name of the respective sequence (see ./res/data)    
```save_pkl```: path to output directory for final results (pickle)  (default: False)  
```eval_mode```: Mode of evaluation out of ```["train", "test", "all"]``` (default : "train")

```
eval.run(
    benchmark_name = benchmark_name,
    seq_file = seq_file,
    gt_dir = gt_dir,
    res_dir = res_dir
        )
```
## Visualization
To visualize your results or the annotations run
<code>
python MOTS/MOTSVisualization.py
</code>

Inside the script adjust the following values for the ```MOTSVisualizer``` class:

```seqName```: Name of the sequence  
```FilePath```: Data file  
```image_dir```: Directory containing images  
```mode```: Video mode. Options: ```None``` for method results, ```raw``` for data video only, and ```gt``` for annotations  
```output_dir```: Directory for created video and thumbnail images  

Additionally, adjust the following values for the ```generateVideo``` function:

```displayTime```: If true, display frame number (default false)  
```displayName```: Name of the method  
```showOccluder```: If true, show occluder of gt data  
```fps```: Frame rate  

```
visualizer = MOTSVisualizer(seqName, FilePath, image_dir, mode, output_dir )
visualizer.generateVideo(displayTime, displayName, showOccluder, fps  )
```

## Data Format

Each line of an annotation txt file is structured like this (where rle means run-length encoding from COCO):
```
time_frame id class_id img_height img_width rle
```
An example line from a txt file:
```
52 1005 1 375 1242 WSV:2d;1O10000O10000O1O100O100O1O100O1000000000000000O100O102N5K00O1O1N2O110OO2O001O1NTga3
```
Meaning:
<br>time frame 52
<br>object id 1005 (meaning class id is 1, i.e. car and instance id is 5)
<br>class id 1
<br>image height 375
<br>image width 1242
<br>rle WSV:2d;1O10000O10000O1O100O100O1O100O1000000000000000O100O...1O1N </p>

image height, image width, and rle can be used together to decode a mask using [cocotools](https://github.com/cocodataset/cocoapi).

## Citation
If you work with the code and the benchmark, please cite:

```
@inproceedings{Voigtlaender19CVPR_MOTS,
 author = {Paul Voigtlaender and Michael Krause and Aljo\u{s}a O\u{s}ep and Jonathon Luiten and Berin Balachandar Gnana Sekar and Andreas Geiger and Bastian Leibe},
 title = {{MOTS}: Multi-Object Tracking and Segmentation},
 booktitle = {CVPR},
 year = {2019},
}
```

## License
MIT License

## Contact
If you find a problem in the code, please open an issue.

For general questions, please contact Paul Voigtlaender (voigtlaender@vision.rwth-aachen.de) or Michael Krause (michael.krause@rwth-aachen.de)
