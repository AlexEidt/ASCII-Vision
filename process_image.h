// Copyright Alex Eidt 2021

#ifndef PROCESS_IMAGE_H_
#define PROCESS_IMAGE_H_

#include <stdint.h>     // uint8_t
#include <stdbool.h>    // bool

#define NUM_CHANNELS 3
#define OUTLINE 1
#define SHARPEN 2
#define EMBOSS 3
#define SOBEL 4

// Processes incoming frames by resizing them and converting to grayscale if specified.
// Parameters:
//      input: Input image as integer array.
//      output: Output image (resized) as integer array.
//      w: Original Image width.
//      h: Original Image height.
//      factor: Factor to resize by.
//      grayscale: If 0, use weighted RGB to compute grayscale.
//                 If 1, use three way max to compute grayscale.
//                 If 2, show image in color.
//      mirror: If True, flip image around vertical axis. For use with webcam stream.
void process_stream(
    uint8_t* input,
    uint8_t* output,
    int w,
    int h,
    int factor,
    int grayscale,
    bool mirror
);

// Finds maximum of three values a, b and c and returns it.
// Parameters:
//      a, b, c: Integers to compare.
// Returns:
//      Maximum of a, b and c.
//
// Originally from Joseph Redmon's repo:
// https://github.com/pjreddie/uwnet
uint8_t three_way_max(uint8_t a, uint8_t b, uint8_t c);

// Maps pixel values to 0-93 ascii representation.
// Parameters:
//      inout: Image to apply ascii filter to.
//      size: Number of pixels in image.
void ascii(uint8_t* inout, int size);

// Applies a convolution to an image specified by kernel_type.
// Parameters:
//      input: Input image as integer array.
//      output: Output image (resized) as integer array.
//      w: Original Image width.
//      h: Original Image height.
//      kernel_type: Kernel enum.
void apply(uint8_t* input, uint8_t* output, int w, int h, int kernel_type);

// Applies a convolution with a given kernel to an image.
// Parameters:
//      input: Input image as integer array.
//      output: Output image (resized) as integer array.
//      kernel: Convolution kernel.
//      w: Original Image width.
//      h: Original Image height.
void convolve(
    uint8_t* input,
    uint8_t* output,
    const uint8_t* kernel,
    int w,
    int h
);

// Calculates integer square root.
// Parameter:
//      n: Integer to take square root of.
// Returns:
//      The integer square root of n.
int isqrt(int n);

// Applies the Sobel Edge Detector to an image.
// Parameters:
//      input: Input image as integer array.
//      output: Output image (resized) as integer array.
//      w: Original Image width.
//      h: Original Image height.
void sobel(uint8_t* input, uint8_t* output, int w, int h);

#endif      // PROCESS_IMAGE_H_