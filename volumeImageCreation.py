import os
import sys

import traceback

import math

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
    # Argument parsing
    #
    
    import argparse
    
    
    __parser = argparse.ArgumentParser(
        description='Creates a volumetric image, properly oriented in space, from a series of images taken from different planes through the imaging volume.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    __parser.add_argument('--plane-images', dest='plane_images', nargs='+', type=str, required=True, metavar='<file name>', help='List of planar images through an imaging volume in the uncompressed TIFF image format. Their order and orientation is specified in the following arguments.')
    __parser.add_argument('--plane-image-order', dest='image_order', default='I', choices=['R', 'L', 'A', 'P', 'I', 'S'], type=str, required=True, metavar='<order>', help='Order of the imaging planes in terms of the anatomical direction. Note that this direction specifies the origin rather than the target. Hence, if the imaging planes are ordered from inferior to superior, the order will be \'I\'.')
    __parser.add_argument('--plane-orientation', dest='plane_orientation', nargs=2, default=['A', 'L'], choices=['R', 'L', 'A', 'P', 'I', 'S'], type=str, required=True, metavar='<orientation>', help='Orientation of the imaging planes in terms of the anatomical directions of the first (vertical) and the second (horizontal) axis. Note that an image\'s origin is in the upper left corner and that these directions specify the origins rather than the targets. Hence, if the axes go from anterior to posterior and left to right, their orientation will be \'A\' and \'L\', respectively.')
    __parser.add_argument('--plane-rotation', dest='plane_rotation', default=0, type=float, required=False, metavar='<angle>', help='Angle of an in-plane rotation required to be added to fully align the image with its specified orientation. The angle is measured in degrees, positively in counter-clockwise direction and negatively in clockwise direction.')
    __parser.add_argument('--plane-spacing', dest='plane_spacing', nargs=2, default=[1., 1.], type=float, required=True, metavar='<spacing>', help='Spacing of the imaging planes along the first (vertical) and the second (horizontal) axis.')
    __parser.add_argument('--plane-height', dest='plane_height', default=1., type=float, required=True, metavar='<height>', help='Height of the imaging planes or, equivalently, the distance between imaging planes.')
    __parser.add_argument('--as-mask', dest='as_mask', action='store_true', required=False, help='Indicates whether to create this volume image as a mask. In this case, the image intesities will be normalised and afterwards a threshold applied to produce a binary mask.')
    __parser.add_argument('--output-file', dest='output_file', type=str, required=True, metavar='<path>', help='Path for the output file. If the files does already exist, it will not be overwritten and rather an error thrown.')
    
    
    
    kwargs = vars(__parser.parse_args())

    # ********************************************************************************
    # Preprocess arguments

    kwargs['plane_rotation'] = math.radians(kwargs['plane_rotation'])

    # ********************************************************************************
    # Execute main function

    try:
        volume_image_creation(**kwargs)
    except:
        print('', file=sys.stdout)
        print('', file=sys.stdout)
        print('An error occured. Operation could not be completed.', file=sys.stdout)
        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1)

    sys.exit(0)
