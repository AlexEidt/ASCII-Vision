# ASCII Vision

Convert video/webcam streams into high resolution ASCII streams with other optional video effects.

## Demo

### Original Video

<img src="Documentation/fireworks.gif" alt="Original Fireworks">

### ASCII Video (Mirrored)

<img src="Documentation/ASCII.gif" alt="Fireworks with ASCII Filter">

## Key Bindings

Key | Description
--- | ---
`G` | Toggle Grayscale and Color Mode
`A` | Toggle ASCII Mode
`O` | Apply Outline Convolution Kernel
`S` | Apply Sobel Filter
`SPACE` | Remove all filters
`1-9` | Change size of ASCII Image
`0` | Reset ASCII Image to Original Size

## Options

While the ASCII Streamer will default to displaying your webcam feed, it can also display video from other sources as well as video files. See the top of `app.py` to see all the options:

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

If you want to stream video from something other than your webcam, go down to line 27 in `app.py`

```python
STREAM = '<video0>'
```

and change the `0` in `'<video0>'` to some integer. It will usually be 1, but if you have several video streaming devices connected, you might have to go through several integers (i.e. 2, 3, 4, etc). You can also change this to be a video file name.

## Dependencies

* Python 3.8+
* `imageio`
* `imageio-ffmpeg`
* `PIL`
* `numpy`
* `keyboard`

```
pip install pillow numpy imageio keyboard
pip install --user imageio-ffmpeg
```

## Acknowledgements

I'm definitely not the first person to make an ASCII converter, and there were definitely some resources that helped me complete this project. They are listed below.

* [Ryan Delaney](https://github.com/Vitineth?tab=followers) made an [Image to ASCII Art Generator](https://github.com/Vitineth/ascii-art-generator) from which I used several functions to sort all ASCII characters in order of their density. While he converted to HCL from RGB to encode each pixel as an ASCII character, I just used the grayscale value. His project is really cool, so make sure to check it out!
* [This blog post](https://www.codespeedy.com/video-streaming-in-tkinter-with-python/) by Satyam Singh Niranjan explained how to display the webcam stream on the tKinter UI in Python.
* The royalty-free fireworks video is from https://www.youtube.com/watch?v=PEYq34x83Xs