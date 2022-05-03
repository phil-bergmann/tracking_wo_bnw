# 3D-ZeF20 - Zebrafish challenge
![3DZeF_PIC](https://motchallenge.net/sequenceVideos/ZebraFish-04-gt.jpg)

## Requirements
* Python 3.6.9
* install [requirements.txt](requirements.txt)
* Note: The tracking evaluation relies on the Python implementation: https://github.com/cheind/py-motmetrics
## Usage


1) Run
```
python ZF3D/evalZF3D.py
```


## Evaluation
To run the evaluation for your method please adjust the file ```ZF3D/evalZF3D.py``` using the following arguments:


```benchmark_name```: Name of the benchmark, e.g. ZeF-3D    
```gt_dir```: Directory containing ground truth files in ```<gt_dir>/<sequence>/gt/gt.txt```    
```res_dir```: The folder containing the tracking results. Each one should be saved in a separate .txt file with the name of the respective sequence (see ./res/data)    
```save_pkl```: path to output directory for final results (pickle)  (default: False)  
```eval_mode```: Mode of evaluation out of ```["train", "test", "all"]``` (default : "train")

```
eval.run(
    benchmark_name = benchmark_name,
    gt_dir = gt_dir,
    res_dir = res_dir,
    eval_mode = eval_mode)
```
## Visualization
To visualize your results or the annotations run
<code>
python ZF3D/ZF3DVisualization.py
</code>

Inside the script adjust the following values for the ```ZF3DVisualizer``` class:

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
visualizer = ZF3DVisualizer(seqName, FilePath, image_dir, mode, output_dir )
visualizer.generateVideo(displayTime, displayName, showOccluder, fps  )
```
## Data Format
Submit your tracking result where each row of your submission file has to contain the following values. The values are defined as in the annotation file, and any other values will be ignored.  
Each line of an annotation txt file is structured as follows:
<pre>
frame: The video frame which the annotation is associated with 
id: Identity of the fish
3d_x: x coordinate of 3D head position in world coordinates
3d_y: y coordinate of 3D head position in world coordinates
3d_z: z coordinate of 3D head position in world coordinates
</pre>

<p>Four example lines of a submission txt file:<p/>
<pre>
1, 1, 19.61, 28.313, 7.93
1, 2, 18.317, 28.636, 8.911
2, 1, 19.685, 28.348, 7.886
2, 2, 18.197, 28.625, 8.868
</pre>


## Citation
If you work with the code and the benchmark, please cite:

```
@article{3DZeF20,
    title={3D-ZeF: A 3D Zebrafish Tracking Benchmark Dataset},
    shorttitle = {3DZeF20},
	url = {https://arxiv.org/abs/2006.08466},
	journal = {arXiv:2006.08466[cs]},
	author={Malte Pedersen and Joakim Bruslund Haurum and Stefan Hein Bengtson and Thomas B. Moeslund},
	year = {2020},
	note = {arXiv: 2006.08466},
	keywords = {Computer Science - Computer Vision and Pattern Recognition}

}
```

## Contact
If you find a problem in the code, please open an issue.

For general questions, please contact Joakim Bruslund (joha@create.aau.dk) or Malte Pedersen (mape@create.aau.dk)
