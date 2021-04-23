# ASCII Streamer

Convert video/webcam streams into high resolution ASCII streams with other optional video effects.

## App

<img src="Documentation/App-Demo.gif"/>

## Demo

### Original

<img src="Documentation/fireworks.gif" alt="Original Fireworks">

### ASCII

<img src="Documentation/ASCII.gif" alt="Fireworks with ASCII Filter">

## Filters

There are several filters to apply to the video stream, all done using convolution kernels. They are the standard Outline, Sharpen, Emboss and Sobel kernels.

## Options

While the ASCII Streamer will default to displaying your webcam feed, it can also display video from other sources as well as video files. Go to lines 22 - 29 in `app.py` to see all the options:

```python
# Mirror image stream along vertical axis.
MIRROR = True
# Video Stream to use. Put filename here if using video file.
STREAM = '<video0>'
# Background color of the ASCII stream.
BACKGROUND_COLOR = 'white'
# Font color used in the ASCII stream. Make sure there's some contrast between the two.
FONT_COLOR = 'black'
```

### Video from Other Sources

If you want to stream video from something other than your webcam, go down to line 25 in `app.py`

```python
STREAM = '<video0>'
```

and change the `0` in `'<video0>'` to some integer. It will usually be 1, but if you have several video streaming devices connected, you might have to go through several integers (i.e. 2, 3, 4, etc).

### Video Files

If you want to play video files as the stream, go to line 25 in `app.py` and change `'<video0>'` in

```python
STREAM = '<video0>'
```

to the file name/path of the video file you want to stream.

## Usage

Before you begin you'll need to compile `process_image.c` into a shared object file. **If you're running 32-bit Python, use a C compiler that compiles to 32-bit. If you're running 64-bit Python, use a C compiler that compiles to 64-bit. If you don't do this, you'll get an error saying your .so file can't be found**.

```
gcc -fPIC -shared -o process_image.so process_image.c
```

If you want to use a different `.so` file name other than `process_image.so`, go to line 14 in `app.py` and enter the new file name there:

```python
lib = CDLL(os.path.join(os.getcwd(), 'process_image.so'), RTLD_GLOBAL)
```

Now that you've compiled the file, run `app.py` and you're good to go!

```
python app.py
```

## Resizing the ASCII Video Stream

If you want to resize the ASCII stream, go to lines 103 - 105 in `app.py` and modify line 104 as shown below.

```python
ascii_label.config(
    text='\n'.join([''.join(x) for x in char_mapper(output).reshape((h, w))])
)

n = 3
ascii_label.config(
    text='\n'.join([''.join(x) for i, x in enumerate(char_mapper(output).reshape((h, w))) if i % n])
)
```

We've added `enumerate` to only sample (n - 1) out of every n lines to make the ASCII image a bit smaller. There are definitely more efficient ways to implement this on the backend, this is just a quick and simple fix.

## Dependencies

* Python 3.8+
* `imageio`
* `imageio-ffmpeg`
* `PIL`
* `numpy`
* C Compiler

```
pip install --user imageio imageio-ffmpeg
pip install pillow
pip install numpy
```

## Contributing

Contributions are welcome! Specifically, some things I'm looking to implement are:

* Colorful ASCII Streams by changing the color of each letter.
* Pause/Play and other video playing buttons to better manage the video stream.
* Better resize function (without using floating point arithmetic).
* Recording/Exporting ASCII video.
* Image Compatability (currently only works with videos and streams).
* A better UI in general (although simplicity should guide your approach).
* Any other cool ideas you may have!

To contribute, fork the repo and implement your changes, and submit a pull request.

## Acknowledgements

I'm definitely not the first person to make an ASCII converter, and there were definitely some resources that helped me complete this project. They are listed below.

* [Ryan Delaney](https://github.com/Vitineth?tab=followers) made an [Image to ASCII Art Generator](https://github.com/Vitineth/ascii-art-generator) from which I used several functions to sort all ASCII characters in order of their density. While he converted to HCL from RGB to encode each pixel as an ASCII character, I just used the grayscale value. His project is really cool, so make sure to check it out!
* [This blog post](https://www.codespeedy.com/video-streaming-in-tkinter-with-python/) by Satyam Singh Niranjan explained how to display the webcam stream on the tKinter UI in Python.
* [This Stack Overflow thread](https://stackoverflow.com/questions/596216/formula-to-determine-perceived-brightness-of-rgb-color) on computationally efficient ways of converting RGB images to grayscale.
* The royalty-free fireworks video is from https://www.youtube.com/watch?v=PEYq34x83Xs