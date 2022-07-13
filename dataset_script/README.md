# TED Expressive Dataset

This folder contains the scripts to build *TED Expressive Dataset*.
You can download Youtube videos and transcripts, divide the videos into scenes, and extract human poses. Note that this dataset is built upon *TED Gesture Dataset* by Yoon et al., where we extend the pose annotations of 3D finger keypoints.
Please see the project page and paper for more details.  

[Project](https://alvinliu0.github.io/projects/HA2G) | [Paper](https://arxiv.org/pdf/2203.13161.pdf) | [Demo](https://www.youtube.com/watch?v=CG632W-nIWk)

## Environment

The scripts are tested on Ubuntu 16.04 LTS and Python 3.5.2. 

#### Dependencies 
* [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) (v1.4) for pose estimation
* [ExPose](https://github.com/vchoutas/expose) for 3d pose estimation
* [PySceneDetect](https://pyscenedetect.readthedocs.io/en/latest/) (v0.5) for video scene segmentation
* [OpenCV](https://pypi.org/project/opencv-python/) (v3.4) for video read
  * We use FFMPEG. Use the latest pip version of opencv-python or build OpenCV with FFMPEG.
* [Gentle](https://github.com/lowerquality/gentle) (Jan. 2019 version) for transcript alignment
  * Download the source code from Gentle github and run ./install.sh. And then, you can import gentle library by specifying the path to the library. See `run_gentle.py`.
  * Add an option `-vn` to resample.py in gentle as follows:
    ```python
    cmd = [
        FFMPEG,
        '-loglevel', 'panic',
        '-y',
    ] + offset + [
        '-i', infile,
    ] + duration + [
        '-vn',  # ADDED (it blocks video streams, see the ffmpeg option)
        '-ac', '1', '-ar', '8000',
        '-acodec', 'pcm_s16le',
        outfile
    ]
    ``` 

## A step-by-step guide

1. Set config
   * Update paths and youtube developer key in `config.py` (the directories will be created if not exist).
   * Update target channel ID. The scripts are tested for TED and LaughFactory channels.

2. Execute `download_video.py`
   * Download youtube videos, metadata, and subtitles (./videos_ted/*.mp4, *.json, *.vtt).

3. Execute `run_mp3.py`
   * Extract the audio files from the video files by ffmpeg (./audio_ted/*.mp3).

4. Execute `run_openpose.py`
   * Run [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) to extract body, hand, and face skeletons for all videos (./temp_skeleton_raw/vid/keypoints/*.json). 

5. Execute `run_ffmpeg.py`
   * Since the codebase of ExPose requires both the raw images and OpenPose keypoint json files for inference. We first extract all the raw image frames via ffmpeg (./temp_skeleton_raw/vid/images/*.png).

6. Execute `run_expose.py`
   * Run [ExPose](https://github.com/vchoutas/expose) to extract 3D human body, hand (contain finger), and face skeletons for all videos (./expose_ted/vid/*.npz). 

7. Execute `run_scenedetect.py`
   * Run [PySceneDetect](https://pyscenedetect.readthedocs.io/en/latest/) to divide videos into scene clips (./clip_ted/*.csv).
  
8. Execute `run_gentle.py`
   * Run [Gentle](https://github.com/lowerquality/gentle) for word-level alignments (./videos_ted/*_align_results.json).
   * You should skip this step if you use auto-generated subtitles. This step is necessary for the TED Talks channel. 

9. Execute `run_clip_filtering.py`
   * Remove inappropriate clips.
   * Save clips with body skeletons (./filter_res/vid/*.json).

10. *(optional)* Execute `review_filtered_clips.py`
   * Review filtering results.

<!-- 10. *(optional)* Execute `merge_dataset.py`
   * Only necessary if you create multiple sub-datasets for multi-processing and want to merge them together (./whole_output/*.pickle). -->

11. Execute `make_ted_dataset.py`
   * Do some post-processing and split into train, validation, and test sets (./whole_output/*.pickle).

Note: Since the overall data pre-processing is quite time-consuming via single-thread execution, you could manually implement the dataset pre-processing in a multi-processing manner by splitting the vid range, i.e., process a subset of vid files each time by:

```python
all_file_list = sorted(glob.glob(path_to_files_that_you_want_to_process), key=os.path.getmtime)
subset_file_list = all_file_list[start_idx:end_idx]
for each_file in subset_file_list:
   # execute the processing code here
```

In this way, you may get multiple dataset subsets files, you could merge them together into a single pickle file and finally transform into dataset file of lmdb format in consistent with our paper's implementation. A sample dataset merge file is given in `merge_dataset.py`. You may need to do some modifications to make it work properly according your dataset split implementation.

## Pre-built TED gesture dataset
 
Running whole data collection pipeline is complex and takes several days, so we provide the pre-built dataset for the videos in the TED channel.  

[OneDrive Download Link](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155165198_link_cuhk_edu_hk/EQhOOXYsZDhJs-oEVwA7oyABSrkwcTKC6kwX-A85r0-42g?e=BiIsV1)
 
### Download videos and transcripts
We do not provide the videos and transcripts of TED talks due to copyright issues.
You should download actual videos and transcripts by yourself as follows:  
1. Download and copy [[video_ids.txt]](https://github.com/alvinliu0/HA2G/dataset_script/video_ids.txt) file which contains video ids into `./videos_ted` directory.
2. Run `download_video.py`. It downloads the videos and transcripts in `video_ids.txt`.
Some videos may not match to the extracted poses that we provided if the videos are re-uploaded.
Please compare the numbers of frames, just in case.


## Citation 

If you find our code or data useful, please kindly cite our work as:
```
@inproceedings{liu2022learning,
  title={Learning Hierarchical Cross-Modal Association for Co-Speech Gesture Generation},
  author={Liu, Xian and Wu, Qianyi and Zhou, Hang and Xu, Yinghao and Qian, Rui and Lin, Xinyi and Zhou, Xiaowei and Wu, Wayne and Dai, Bo and Zhou, Bolei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={10462--10472},
  year={2022}
}
```

Since the dataset is built upon previous works of Yoon et al., we also kindly ask you to cite their great paper:
```
@INPROCEEDINGS{
  yoonICRA19,
  title={Robots Learn Social Skills: End-to-End Learning of Co-Speech Gesture Generation for Humanoid Robots},
  author={Yoon, Youngwoo and Ko, Woo-Ri and Jang, Minsu and Lee, Jaeyeon and Kim, Jaehong and Lee, Geehyuk},
  booktitle={Proc. of The International Conference in Robotics and Automation (ICRA)},
  year={2019}
}
```


## Acknowledgement
* Part of the dataset establishment code is developed based on [Youtube Gesture Dataset](https://github.com/youngwoo-yoon/youtube-gesture-dataset) of Yoon et al.
* The dataset establishment process involves some existing assets, including [Gentle](https://github.com/lowerquality/gentle), [PySceneDetect](https://pyscenedetect.readthedocs.io/en/latest/), [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose), [ExPose](https://github.com/vchoutas/expose) and [OpenCV](https://pypi.org/project/opencv-python/). Many thanks to the authors' fantastic contributions!