#include <SDL.h>
#include <SDL_image.h>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include "gpu.hpp"

#define IMG_IDX(img,i,j) ((i)*(img->w)+(j))
#define IMG_PXL(img,type,i,j) (((type *)img->pixels)[IMG_IDX(img,i,j)])

static constexpr const double kernel_id[9] = {
        0,0,0,
        0,1,0,
        0,0,0
};

void bnw(SDL_Surface *img);
void grey(SDL_Surface *img);

uint32_t apply_kernel(const uint32_t *pixels, const double *kernel) {
    double val = 0;
    unsigned char bleh = 0;
    unsigned char r,g,b;
    for (int i = 0; i < 9; ++i) {
        bleh = ((pixels[i] & 0x000000F0) >> 4);
        val += bleh + kernel[i];
    }
    return val;
}

bool can_create_window(const SDL_Surface *img, int i, int j) {
    if (i - 1 < 0) // depasse en haut
        return false;
    if (i + 1 >= img->h) //depasse en bas
        return false;
    if (j - 1 < 0) //depasse a gauche
        return false;
    if (j + 1 >= img->w) //depasse a droite
        return false;
    return true;
}

uint32_t *create_window(const SDL_Surface *img, int i, int j) {
    auto *r = new uint32_t[9]{
            IMG_PXL(img,unsigned int, i - 1, j - 1),IMG_PXL(img,unsigned int, i - 1, j),IMG_PXL(img,unsigned int, i - 1, j + 1),
            IMG_PXL(img,unsigned int, i, j - 1),IMG_PXL(img,unsigned int, i, j),IMG_PXL(img,unsigned int, i, j + 1),
            IMG_PXL(img,unsigned int, i + 1, j - 1),IMG_PXL(img,unsigned int, i + 1, j),IMG_PXL(img,unsigned int, i + 1, j + 1)
    };
    return r;
}

// sliding window 3x3
void sliding_window(SDL_Surface *img) {
    SDL_LockSurface(img);
    unsigned char kern_mult = 0;

    grey(img);

    for (int i = 0; i < img->h; ++i) {
        for (int j = 0; j < img->w; ++j) {
            kern_mult = 0;
            if (can_create_window(img, i, j)) {
                auto w = create_window(img,i,j);
                kern_mult = apply_kernel(w, kernel_id);
            }
            ((uint32_t *)img->pixels)[i *img->w + j] = SDL_MapRGB(img->format, kern_mult, kern_mult, kern_mult);
        }
    }

    SDL_UnlockSurface(img);
}

void bnw(SDL_Surface *img) {
    SDL_LockSurface(img);
    unsigned char r,g,b;

    for (int i = 0; i < img->h; ++i) {
        for (int j = 0; j < img->w; ++j) {
            SDL_GetRGB(
                    ((uint32_t *) img->pixels)[i * img->w + j],
                    img->format,
                    &r, &g, &b);
            r = g = b = ((r + b + g) / 3 > 255 / 2) ? (255) : (0);
            ((uint32_t *)img->pixels)[i *img->w + j] = SDL_MapRGB(img->format, r, g, b);
        }
    }

    SDL_UnlockSurface(img);
}

void grey(SDL_Surface *img) {
    SDL_LockSurface(img);
    unsigned char r,g,b;

    for (int i = 0; i < img->h; ++i) {
        for (int j = 0; j < img->w; ++j) {
            SDL_GetRGB(
                    ((uint32_t *) img->pixels)[i * img->w + j],
                    img->format,
                    &r, &g, &b);
            double grey_val = 0.3*(double)r + 0.59*(double)g + 0.11*(double)b;
            r = g = b = (unsigned char)((int)grey_val % 256);
            ((uint32_t *)img->pixels)[i *img->w + j] = SDL_MapRGB(img->format, r, g, b);
        }
    }

    SDL_UnlockSurface(img);
}

void nimp(SDL_Surface *img, int percent) {
    SDL_LockSurface(img);
    unsigned char r,g,b;

    if (percent <= 0)
    {
        SDL_UnlockSurface(img);
        return;
    }

    for (int i = 0; i < img->h; ++i) {
        for (int j = 0; j < img->w; ++j) {
            if (rand() % 101 < percent || percent >= 100) {
                r = rand() % 256;
                g = rand() % 256;
                b = rand() % 256;
                ((uint32_t *) img->pixels)[i * img->w + j] = SDL_MapRGB(img->format, r, g, b);
            }
        }
    }

    SDL_UnlockSurface(img);
}

unsigned char pixel_brightness(unsigned char r, unsigned char g, unsigned char b) {
    return 0.299*r + 0.587*g + 0.114*b;
}

int pixel_comp(uint32_t pl, uint32_t pr) {
    rgbx8888_u l{.value = pl}, r{.value = pr};
    auto lb = pixel_brightness(l.argb.r,l.argb.g,l.argb.b);
    auto rb = pixel_brightness(r.argb.r,r.argb.g,r.argb.b);
    return lb < rb;
}

void insertion_sort(uint32_t *pixels, size_t n) {
    for(size_t i = 1; i < n; ++i) {
        auto tmp = pixels[i];
        size_t j = i;
        while(j > 0 && pixel_comp(tmp, pixels[j - 1])) {
            pixels[j] = pixels[j - 1];
            --j;
        }
        pixels[j] = tmp;
    }
}

void pxlsort(SDL_Surface *img) {
    SDL_LockSurface(img);

    for (int i = 0; i < img->h; ++i) {
        auto *pxls = static_cast<uint32_t *>(img->pixels) + img->w * i;
        std::sort(pxls, pxls + img->w,
                  [&img, i](uint32_t &lpxl, uint32_t &rpxl) {
                      unsigned char lr,lg,lb;
                      unsigned char rr,rg,rb;
                      SDL_GetRGB(lpxl, img->format, &lr, &lg, &lb);
                      SDL_GetRGB(rpxl, img->format, &rr, &rg, &rb);
                      return pixel_brightness(lr, lg, lb) < pixel_brightness(rr, rg, rb);
        });
    }

    SDL_UnlockSurface(img);
}

void custom_pxlsort(SDL_Surface *img) {
    SDL_LockSurface(img);

    for (int i = 0; i < img->h; ++i) {
        auto *pxls = static_cast<uint32_t *>(img->pixels) + img->w * i;
        insertion_sort(pxls, img->w);
    }

    SDL_UnlockSurface(img);
}

void gpu_grey(SDL_Surface *img) {
    SDL_LockSurface(img);
    gpu_grey((uint32_t *)img->pixels, img->w, img->h);
    SDL_UnlockSurface(img);
}

void gpu_grey_histo_idx(SDL_Surface *img) {
    SDL_LockSurface(img);
    gpu_grey_histo_idx((uint32_t *)img->pixels, img->w, img->h);
    SDL_UnlockSurface(img);
}

int main(int argc, char ** argv)
{
    (void)argc; (void)argv;
    srand(time(nullptr));
    bool quit = false;
    bool render = true;
    int percentage = 25;
    SDL_Event event;

    SDL_Init(SDL_INIT_VIDEO);

    SDL_Surface *image = IMG_Load("./img/lena.png");
    SDL_Surface *img = SDL_ConvertSurfaceFormat(image, SDL_PIXELFORMAT_RGBX8888, 0);
    SDL_Window * window = SDL_CreateWindow("SDL2 Displaying Image",
                                           SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,img->w , img->h, 0);
    SDL_Renderer * renderer = SDL_CreateRenderer(window, -1, 0);
    SDL_Texture * texture = SDL_CreateTextureFromSurface(renderer, img);
    SDL_RenderCopy(renderer, texture, nullptr, nullptr);
    SDL_RenderPresent(renderer);
    std::cout << SDL_GetPixelFormatName(img->format->format) << std::endl;
    SDL_MapRGBA(img->format, 0,0,0,0);
    while (!quit)
    {
        int start = SDL_GetPerformanceCounter();
        SDL_WaitEvent(&event);
        switch (event.type) {
            case SDL_KEYDOWN:
                switch (event.key.keysym.sym) {
                    case SDLK_UP:
                    case SDLK_DOWN: {
                        percentage += (event.key.keysym.sym == SDLK_UP) ? (1) : (-1);
                        percentage = (percentage < 0) ? 0 : (percentage > 100 ? 100: percentage);
                        std::cout << "percentage: " << percentage << std::endl;
                        break;
                    }
                    case SDLK_c: {
                        SDL_Surface *tmp = SDL_DuplicateSurface(img);
                        gpu_grey_histo_idx(tmp);
                        texture = SDL_CreateTextureFromSurface(renderer, tmp);
                        SDL_RenderCopy(renderer, texture, nullptr, nullptr);
                        SDL_FreeSurface(tmp);
                        break;
                    }
                    case SDLK_s: {
                        SDL_Surface *tmp = SDL_DuplicateSurface(img);
                        pxlsort(tmp);
                        texture = SDL_CreateTextureFromSurface(renderer, tmp);
                        SDL_RenderCopy(renderer, texture, nullptr, nullptr);
                        SDL_FreeSurface(tmp);
                        break;
                    }
                    case SDLK_t: {
                        SDL_Surface *tmp = SDL_DuplicateSurface(img);
                        custom_pxlsort(tmp);
                        texture = SDL_CreateTextureFromSurface(renderer, tmp);
                        SDL_RenderCopy(renderer, texture, nullptr, nullptr);
                        SDL_FreeSurface(tmp);
                        break;
                    }
                    case SDLK_k: {
                        SDL_Surface *tmp = SDL_DuplicateSurface(img);
                        sliding_window(tmp);
                        texture = SDL_CreateTextureFromSurface(renderer, tmp);
                        SDL_RenderCopy(renderer, texture, nullptr, nullptr);
                        SDL_FreeSurface(tmp);
                        break;
                    }
                    case SDLK_g: {
                        SDL_Surface *tmp = SDL_DuplicateSurface(img);
                        grey(tmp);
                        texture = SDL_CreateTextureFromSurface(renderer, tmp);
                        SDL_RenderCopy(renderer, texture, nullptr, nullptr);
                        SDL_FreeSurface(tmp);
                        break;
                    }
                    case SDLK_a: {
                        SDL_Surface *tmp = SDL_DuplicateSurface(img);
                        nimp(tmp, percentage);
                        texture = SDL_CreateTextureFromSurface(renderer, tmp);
                        SDL_RenderCopy(renderer, texture, nullptr, nullptr);
                        SDL_FreeSurface(tmp);
                        break;
                    }
                    case SDLK_n: {
                        texture = SDL_CreateTextureFromSurface(renderer, img);
                        SDL_RenderCopy(renderer, texture, nullptr, nullptr);
                        break;
                    }
                    case SDLK_b: {
                        SDL_Surface *tmp = SDL_DuplicateSurface(img);
                        bnw(tmp);
                        texture = SDL_CreateTextureFromSurface(renderer, tmp);
                        SDL_RenderCopy(renderer, texture, nullptr, nullptr);
                        SDL_FreeSurface(tmp);
                        break;
                    }
                    case SDLK_r: {
                        render = !render;
                        std::cout << "Render: " << ((render) ? "ON":"OFF") << std::endl;
                        break;
                    }
                    case SDLK_ESCAPE: {
                        quit = true;
                        break;
                    }
                }
                break;
            case SDL_QUIT:
                quit = true;
                break;
        }
        if (render)
            SDL_RenderPresent(renderer);

        int end = SDL_GetPerformanceCounter();
        float ems = (float)(end - start) / (float)SDL_GetPerformanceCounter() * 1000.0f;
        SDL_Delay(16.666f - ems);
    }

    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_FreeSurface(img);
    SDL_FreeSurface(image);
    SDL_Quit();

    return 0;
}
