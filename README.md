# ASCII Vision

Convert video/webcam streams into high resolution ASCII streams with other optional video effects.

Check out the [ASCII-Video](https://github.com/AlexEidt/ASCII-Video) Repository to see how the real time color ASCII stream was achieved!

## Demo

<img src="Documentation/ducks.gif" alt="Ducks Video in ASCII Streamer">

## Key Bindings

Key | Description
--- | ---
`G` | Toggle Grayscale and Color Mode
`A` | Toggle ASCII Mode
`T` | Toggle Text Mode
`O` | Apply Outline Convolution Kernel
`S` | Apply Sobel Filter
`SPACE` | Remove all filters
`1-9` | Change fontsize/size of ASCII Image
`0` | Reset ASCII Image to Original Size
`Shift` | Shift+[KEY] will undo that operation. Example: `Shift+A` removes ASCII mode

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
# Font size to use with colored/grayscaled ASCII
FONTSIZE = 12
# Boldness to use with colored/grayscaled ASCII
BOLDNESS = 1
# Factor to divide image height and width by. 1 For for original size, 2 for half size, etc...
FACTOR = 2
```

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