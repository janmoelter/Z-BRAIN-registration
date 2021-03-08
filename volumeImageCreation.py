import os
import sys

import traceback

import ants
import numpy as np

#import matplotlib.image as img
import tifffile

from Modules import VolumeImagery
    
    
def volume_image_creation(plane_images, image_order='I', plane_orientation=('A', 'L'), plane_rotation=0, plane_spacing=(1.,1.), plane_height=1., as_mask=False, output_file='out.nrrd'):
    
    if not all([os.path.isfile(_) for _ in plane_images]):
        raise FileNotFoundError('At least one plane image cannot be found.')
    
    __image_stack = [tifffile.imread(plane_image).astype('float32') for plane_image in plane_images]
    
    
    __volume_image = VolumeImagery.from_image_stack(__image_stack, stack_orientation=image_order, image_orientation=plane_orientation, image_rotation=plane_rotation, image_spacing=plane_spacing, image_height=plane_height)
    
    
    if as_mask:
        __volume_image = VolumeImagery.normalise(__volume_image, range=(0,1), as_mask=as_mask)
    
    
    if not os.path.splitext(output_file)[1].upper() in ['.NRRD']:
        output_file += '.nrrd'
            
    VolumeImagery.save(__volume_image, output_path=output_file)


if __name__ == "__main__":
    # ********************************************************************************
    # Usage:
    # > python volumeImageCreation.py --plane-images <...>
    #                                 --image-order <...>
    #                                 --plane-orientation <...>
    #                                 --plane-spacing <...>
    #                                 --plane-height <...>
    #                                 --as-mask
    #                                 --output-file <...>
    #
    
    import argparse
    
    
    __parser = argparse.ArgumentParser()
    __parser.add_argument('--plane-images', dest='plane_images', nargs='+', type=str, required=True)
    __parser.add_argument('--image-order', dest='image_order', default='S', type=str, required=False)
    __parser.add_argument('--plane-orientation', dest='plane_orientation', nargs='+', default=['P', 'R'], type=str, required=False)
    __parser.add_argument('--plane-rotation', dest='plane_rotation', default=1., type=float, required=False)
    __parser.add_argument('--plane-spacing', dest='plane_spacing', nargs='+', default=[1., 1.], type=float, required=True)
    __parser.add_argument('--plane-height', dest='plane_height', default=1., type=float, required=False)
    __parser.add_argument('--as-mask', dest='as_mask', action='store_true', required=False)
    __parser.add_argument('--output-file', dest='output_file', type=str, required=True)
    
    
    
    kwargs = vars(__parser.parse_args())

    # ********************************************************************************
    # Include default parameters


    # ********************************************************************************
    # Execute main function

    try:
        volume_image_creation(**kwargs)
    except:
        print('An error occured. Operation could not be completed.', file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1)

    sys.exit(0)
