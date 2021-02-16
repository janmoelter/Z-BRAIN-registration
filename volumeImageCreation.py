import os
import sys

import ants
import numpy as np

#import matplotlib.image as img
import tifffile

from Modules import VolumeImagery
    
    
def volume_image_creation(plane_images, plane_spacing=(1.,1.), plane_height=1., plane_axis=2, plane_rotation=0, reverse_order=False, spacing=None, as_mask=False, output_file='out.nrrd'):
    
    __plane_image_arrays = [tifffile.imread(plane_image).T if os.path.isfile(plane_image) else None for plane_image in plane_images]

    if reverse_order:
        __plane_image_arrays.reverse()

    if not plane_rotation == 0:
        __plane_image_arrays = [np.rot90(_, k=plane_rotation) for _ in __plane_image_arrays]
    
    __volume_image = VolumeImagery.from_image_stack(__plane_image_arrays, image_spacing=plane_spacing, image_height=plane_height, axis=plane_axis, spacing=spacing, binary_mask=as_mask)
    
    
    if not os.path.splitext(output_file)[1].upper() in ['.NRRD']:
        output_file += '.nrrd'
            
    VolumeImagery.save(__volume_image, output_path=output_file)


if __name__ == "__main__":
    # ********************************************************************************
    # Usage:
    # > python volumeImageCreation.py --plane-images <...>
    #                                 --plane-spacing <...>
    #                                 --plane-height <...>
    #                                 --plane-rotation <...>
    #                                 --reverse-order
    #                                 --spacing <...>
    #                                 --as-mask
    #                                 --output-file <...>
    #
    
    import argparse
    
    
    __parser = argparse.ArgumentParser()
    __parser.add_argument('--plane-images', dest='plane_images', nargs='+', type=str, required=True)
    __parser.add_argument('--plane-spacing', dest='plane_spacing', nargs='+', default=[1., 1.], type=float, required=True)
    __parser.add_argument('--plane-height', dest='plane_height', default=1., type=float)
    __parser.add_argument('--plane-axis', dest='plane_axis', default=2, type=int, required=False)
    __parser.add_argument('--plane-rotation', dest='plane_rotation', default=0, type=int, required=False)
    __parser.add_argument('--reverse-order', dest='reverse_order', action='store_true', required=False)
    __parser.add_argument('--spacing', nargs='+', default=None, type=float, required=False)
    __parser.add_argument('--as-mask', dest='as_mask', action='store_true', required=False)
    __parser.add_argument('--output-file', dest='output_file', type=str, required=True)
    
    kwargs = vars(__parser.parse_args())

    # ********************************************************************************
    # Include default parameters

    if len(kwargs['plane_spacing']) == 1:
        kwargs['plane_spacing'] = kwargs['plane_spacing'] * 2
    elif len(kwargs['plane_spacing']) > 2:
        kwargs['plane_spacing'] = kwargs['plane_spacing'][:2]

    if kwargs['spacing'] is not None and len(kwargs['spacing']) in [1, 3]:
        if len(kwargs['spacing']) == 1:
            kwargs['spacing'] = kwargs['spacing'] * 3
    else:
        kwargs['spacing'] = None

    # ********************************************************************************
    # Execute main function

    try:
        volume_image_creation(**kwargs)
    except:
        print('An error occured. Operation could not be completed.', file=sys.stderr)
        sys.exit(1)

    sys.exit(0)
