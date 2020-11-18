//
// Created by dotty on 05/11/2020.
//

#ifndef IMAGE_GPU_HPP
#define IMAGE_GPU_HPP
#include <cstdint>

struct rgbx8888_t {
    unsigned char x; // unused
    unsigned char b;
    unsigned char g;
    unsigned char r;
};

union rgbx8888_u {
    rgbx8888_t argb;
    uint32_t value;
};

void gpu_grey(uint32_t *img, int w, int h);

void gpu_grey_histo_idx(uint32_t *img, int w, int h);

#endif //IMAGE_GPU_HPP
