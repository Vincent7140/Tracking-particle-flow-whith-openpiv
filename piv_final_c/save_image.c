#include <stdio.h>
#include <stdlib.h>
#include <png.h>
#include <string.h>


int save_image(const char* filename, unsigned char* image, int width, int height) {
    FILE *fp = fopen(filename, "wb");
    if(!fp) return -1;

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) return -1;

    png_infop info = png_create_info_struct(png);
    if (!info) return -1;

    if (setjmp(png_jmpbuf(png))) return -1;

    png_init_io(png, fp);

    // Output is 8bit depth, RGBA format.
    png_set_IHDR(
        png,
        info,
        width, height,
        8,
        PNG_COLOR_TYPE_RGBA,
        PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT,
        PNG_FILTER_TYPE_DEFAULT
    );
    png_write_info(png, info);

    png_bytep row = (png_bytep) malloc(4 * width * sizeof(png_byte));
    for(int y = 0; y < height; y++) {
        memcpy(row, &image[y * 4 * width], 4 * width * sizeof(png_byte));
        png_write_row(png, row);
    }

    png_write_end(png, NULL);
    fclose(fp);

    if (png && info)
        png_destroy_write_struct(&png, &info);
    if (row)
        free(row);

    return 0;
}
