import os
import sys

import math

import numpy as np

import tifffile

from Modules import VolumeImagery
    
    
def image_stack_interpolation(stack_images, stack_spacing, target_spacing, output_file_format):
    
    if all([type(image) is str for image in stack_images]):
        __image_stack = [tifffile.imread(image).T if os.path.isfile(image) else None for image in stack_images]
    else:
        __image_stack = stack_images
    
    __interpolated_image_stack = VolumeImagery.interpolate_image_stack(__image_stack, stack_spacing=stack_spacing, target_spacing=target_spacing)


    __counter_length = math.ceil(math.log10(len(__interpolated_image_stack)))

    if not os.path.splitext(output_file_format)[1].upper() in ['.TIF', '.TIFF']:
        output_file_format += '.tif'

    for i in range(len(__interpolated_image_stack)):
        tifffile.imsave(output_file_format.format(('{:0'+ str(__counter_length) +'}').format(i)), __interpolated_image_stack[i].T)
        


if __name__ == "__main__":
    # ********************************************************************************
    # Usage:
    # > python imageStackInterpolation.py --plane-images <...>
    #                                     --plane-distance <...>
    #                                     --interpolation-distance <...>
    #                                     --output-file <...>
    #
    
    import argparse
    
    
    __parser = argparse.ArgumentParser()
    __parser.add_argument('--plane-images', dest='stack_images', nargs='+', type=str, required=True)
    __parser.add_argument('--plane-spacing', dest='stack_spacing', nargs='+', type=float, required=True)
    __parser.add_argument('--interpolation-distance', dest='target_spacing', type=float, required=True)
    __parser.add_argument('--output-file-format', dest='output_file_format', type=str, required=True)
    
    kwargs = vars(__parser.parse_args())

    # ********************************************************************************
    # Include default parameters


    # ********************************************************************************
    # Execute main function

    try:
        image_stack_interpolation(**kwargs)
    except:
        print('An error occured. Operation could not be completed.', file=sys.stderr)
        sys.exit(1)

    sys.exit(0)
