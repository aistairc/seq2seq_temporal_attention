## Requirements (Windows)

+ [Python 2](https://www.continuum.io/downloads)
    (Anaconda is recommend.)
+ [Chainer](http://chainer.org/)  
    In Anaconda Prompt:  
    ```
    pip install chainer
    ```
+ [OpenCV](http://opencv.org/)  
    In Anaconda Prompt:  
    ```
    conda install -c https://conda.anaconda.org/menpo opencv3
    ```
+ [youtube-dl](https://github.com/rg3/youtube-dl/)  
    In Anaconda Prompt:  
    ```
    pip install --upgrade youtube-dl
    ```
+ [Git](https://git-for-windows.github.io/)
+ [wget](http://gnuwin32.sourceforge.net/packages/wget.htm) (Download binary and set path.)
+ [ffmpeg](https://www.ffmpeg.org/) (Download binary and set path.)
+ [unzip](http://gnuwin32.sourceforge.net/packages/unzip.htm) (Download binary and set path.)

Windows cannot use symbolic links,
so please copy `coco-caption/pycocoevalcap` and `coco-caption/pycocotools` to `code/` before running.
