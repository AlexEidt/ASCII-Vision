# Alex Eidt

from tkinter import *
from tkinter.ttk import Style
import imageio
import os
import numpy as np
import keyboard
from PIL import Image, ImageTk


# Mirror image stream along vertical axis.
MIRROR = True
# Video Stream to use. Put filename here if using video file.
STREAM = '<video0>'
# Background color of the ASCII stream.
BACKGROUND_COLOR = 'white'
# Font color used in the ASCII stream. Make sure there's some contrast between the two.
FONT_COLOR = 'black'

COLOR = 1
ASCII = 0
FILTER = 0
BLOCKS = 0

def update():
    """
    Update settings based on user input.
    """
    global COLOR, ASCII, FILTER, BLOCKS

    if keyboard.is_pressed('shift+g'):  # Color/Grayscale Mode
        COLOR = 1
    elif keyboard.is_pressed('g'):
        COLOR = 0
    if keyboard.is_pressed('shift+a'):  # ASCII Mode
        ASCII = 0
    elif keyboard.is_pressed('a'):
        ASCII = 1

    if keyboard.is_pressed('o'):        # Outline Filter
        FILTER = 1
    elif keyboard.is_pressed('s'):      # Sobel Filter 
        FILTER = 2
    elif keyboard.is_pressed('space'):  # No Filter
        FILTER = 0

    for i in range(0, 10):
        if keyboard.is_pressed(str(i)):
            BLOCKS = i
            break


def get_dims(w, h, factor):
    """
    Finds the optimal resizing factor for the webcam stream based on the screen dimensions.
    """
    root = Tk()
    # Get screen size
    height = root.winfo_screenheight() / factor
    width = root.winfo_screenwidth() / factor
    root.destroy()
    scale = None
    max_resolution = 1 << 16
    for i in range(2, 21):
        if not (w % i or h % i):
            max_ = abs(height - h / i) + abs(width - w / i)
            if max_ < max_resolution:
                max_resolution = max_
                scale = i
    return scale


def tile_tuples(w, h):
    """
    Return tile sizes for resizing ASCII Images.
    """
    result = lambda x: [i for i in range(2, x) if x % i == 0]
    return list(zip(result(w), result(h)))


def convolve(frame, kernel):
    """
    Peform a 2D image convolution on the given frame with the given kernel.
    """
    height, width = frame.shape
    kernel_height, kernel_width = kernel.shape
    # assert kh == kw
    output = np.pad(frame, kernel_height // 2, mode='edge')

    output_shape = kernel.shape + tuple(np.subtract(output.shape, kernel.shape) + 1)
    strides = output.strides + output.strides

    return np.einsum(
        'ij,ijkl->kl',
        kernel,
        np.lib.stride_tricks.as_strided(output, output_shape, strides)
    )


def main():
    # All ASCII characters used in the images sorted by pixel density.
    chars = f""" `.,|^'\/~!_-;:)("><¬?*+7j1ilJyc&vt0$VruoI=wzCnY32LTxs4Zkm5hg6qfU9paOS#£eX8D%bdRPGFK@AMQNWHEB"""
    char_mapper = np.vectorize(lambda x: chars[x])

    def stream(scale):
        try:
            image = video.get_next_data()
        except Exception:
            video.close()
            return

        # Update settings based on pressed keys.
        update()

        h, w, c = image.shape

        # ASCII image is larger than regular, so multiply scaling factor by 2 if ASCII mode is on.
        size = scale << 1 if ASCII else scale
        h //= size
        w //= size

        # Resize Image
        image = image[::size, ::size]
        if not COLOR: # Grayscale Image
            image = np.sum(image * np.array([0.299, 0.587, 0.114]), axis=2, dtype=np.uint8)
        if MIRROR: # Mirror Image along vertical axis
            image = np.fliplr(image)

        if BLOCKS > 0 and not COLOR and ASCII:
            dw, dh = TILES[BLOCKS]
            image = (np.add.reduceat(
                np.add.reduceat(image.astype(np.int), np.arange(0, h, dh), axis=0),
                np.arange(0, w, dw),
                axis=1
            ) / (dw * dh)).astype(np.uint8)
            h, w = image.shape

        if FILTER > 0 and not COLOR:
            if FILTER == 1:     # Outline Kernel
                image = convolve(
                    image,
                    np.array([[-1, -1, -1], [-1, -8, -1], [-1, -1, -1]])
                ).astype(np.uint8)
            elif FILTER == 2:   # Sobel Kernel
                gx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
                gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
                image = np.hypot(convolve(image, gx), convolve(image, gy)).astype(np.uint8)
                del gx, gy

        # If ASCII mode is on convert frame to ascii and display, otherwise display video stream.
        if ASCII and not COLOR:
            num_chars = len(chars) - 1
            image = np.delete(image, [i for i in range(h) if not i % 4], axis=0)
            image = num_chars - ((image.astype(np.int) * num_chars) // 255)
            image_label.pack_forget()
            ascii_label.pack()
            # Update label with new ASCII image.
            ascii_label.config(
                text='\n'.join((''.join(x) for x in char_mapper(image))),
                font=('courier', (BLOCKS * 4) + 2)
            )
            ascii_label.after(delay, lambda: stream(scale))
        else:
            ascii_label.pack_forget()
            image_label.pack()
            frame_image = Image.fromarray(image)
            frame_image = ImageTk.PhotoImage(frame_image)
            image_label.config(image=frame_image)
            image_label.image = frame_image
            image_label.after(delay, lambda: stream(scale))

    # Set up window.
    root = Tk()
    root.title('ASCII Streamer')
    f1 = Frame()
    image_label = Label(f1, borderwidth=5, relief='solid')
    ascii_label = Label(f1, font=('courier', 2), fg=FONT_COLOR, bg=BACKGROUND_COLOR, borderwidth=5, relief='solid')
    f1.pack(side=LEFT, expand=YES, padx=10)
    root.protocol("WM_DELETE_WINDOW", lambda: (video.close(), root.destroy()))

    # Get image stream from webcam or other source and begin streaming.
    video = imageio.get_reader(STREAM)
    meta_data = video.get_meta_data()
    # To change the framerate of the stream, change the "delay" value below.
    delay = int(meta_data['fps'])
    h, w, _ = video.get_next_data().shape
    scale = get_dims(w, h, 2)
    if scale is None:
        raise ValueError('Could not find rescaling factor for video/webcam stream.')

    TILES = tile_tuples(w // scale, h // scale)

    stream(scale)
    root.mainloop()


if __name__ == '__main__':
    main()