import os
import sys

import traceback
import warnings

import math

import numpy

import tifffile

from Modules import VolumeImagery
    
    
def volume_image_from_image_stack(plane_images, image_order='I', plane_orientation=('A', 'L'), plane_rotation=0, plane_spacing=(1.,1.), plane_height=1., as_mask=False, output_file='out.nrrd'):
    
    for plane_image in plane_images:
        if not os.path.isfile(plane_image):
            raise FileNotFoundError('Plane image `{}` cannot be found.'.format(plane_image))
    
    __image_stack = [tifffile.imread(plane_image).astype('float32') for plane_image in plane_images]
    
    
    __volume_image = VolumeImagery.from_image_stack(__image_stack, stack_orientation=image_order, image_orientation=plane_orientation, image_rotation=plane_rotation, image_spacing=plane_spacing, image_height=plane_height)
    
    
    if as_mask:
        __volume_image = VolumeImagery.normalise(__volume_image, range=(0,1), as_mask=as_mask)
    
    
    if not os.path.splitext(output_file)[1].upper() in ['.NRRD']:
        output_file += '.nrrd'

    VolumeImagery.save(__volume_image, output_path=output_file)

def volume_image_to_image_stack(volume_image, image_order='I', plane_orientation=('A', 'L'), output_file='out.tif'):

    if not os.path.isfile(volume_image):
        raise FileNotFoundError('Volume image `{}` cannot be found.'.format(volume_image))

    __volume_image = VolumeImagery.load(volume_image, pixeltype='float')


    __NUMPY_INTEGER_TYPE = None

    if numpy.all(numpy.mod(__volume_image[:,:,:], 1) == 0):
        if 0 <= __volume_image.min() and __volume_image.max() < 2**32:
            __volume_image = __volume_image.astype('uint32')

            __NUMPY_INTEGER_TYPE = 'uint32'
            if __volume_image.max() < 2**16:
                __NUMPY_INTEGER_TYPE = 'uint16'
            if __volume_image.max() < 2**8:
                __NUMPY_INTEGER_TYPE = 'uint8'


    __image_stack, __image_stack_spacing = VolumeImagery.to_image_stack(__volume_image, stack_orientation=image_order, image_orientation=plane_orientation, return_spacing=True)

    if __NUMPY_INTEGER_TYPE is not None:
        __image_stack = [_.astype(__NUMPY_INTEGER_TYPE) for _ in __image_stack]

    if not os.path.splitext(output_file)[1].upper() in ['.TIF']:
        output_file += '.tif'

    with tifffile.TiffWriter(output_file, imagej=True) as tif:
        kwargs = {
            'photometric' : 'MINISBLACK',
            #'resolution' : tuple(list(1 / numpy.array(__image_stack_spacing[1:])) + ['MICROMETER']),
            'software' : '',
            'resolution' : tuple(1 / numpy.array(__image_stack_spacing[1:])),
            'metadata' : {
                'spacing' : __image_stack_spacing[0],
                'unit' : 'um',
            },
        }

        for _ in __image_stack:
            tif.write(_, contiguous=True, **kwargs)

def volume_image_resample(volume_image, spacing, output_file='out.nrrd'):

    if not os.path.isfile(volume_image):
        raise FileNotFoundError('Volume image `{}` cannot be found.'.format(volume_image))

    __volume_image = VolumeImagery.load(volume_image, pixeltype=None)
    
    
    __volume_image = VolumeImagery.resample(__volume_image, spacing=spacing, preserve_orientation=True)
    
    
    if not os.path.splitext(output_file)[1].upper() in ['.NRRD']:
        output_file += '.nrrd'

    VolumeImagery.save(__volume_image, output_path=output_file)


if __name__ == "__main__":
    # ********************************************************************************
    # Argument parsing
    #

    import argparse


    __parser = argparse.ArgumentParser(
        description='Either builds a volume image from a stack of plane images taken from different planes through the imaging volume or takes a volume image and slices it into a stack of plane images.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    __subparsers = __parser.add_subparsers(title='subcommands',dest='subcommand', required=True, help='Selection of the interface to build a volume image from a stack of plane images, to slice a volume image into a stack of plane images, or to resample a volume image.')

    __subparser = dict()
    __subparser['from-plane-images'] = __subparsers.add_parser('from-plane-images',
        description='Builds a volume image from a stack of plane images taken from different planes through the imaging volume.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    __subparser['to-plane-images'] = __subparsers.add_parser('to-plane-images',
        description='Slices a volume image into a stack of plane images.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    __subparser['resample'] = __subparsers.add_parser('resample',
        description='Resamples a volume image.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # *** 'from-plane-images' ***

    __subparser['from-plane-images'].add_argument('--plane-images', dest='plane_images', nargs='+', type=str, required=True, metavar='<file name>', help='List of planar images through an imaging volume in the uncompressed TIFF image format. Their order and orientation is specified in the following arguments.')
    __subparser['from-plane-images'].add_argument('--plane-image-order', dest='image_order', default='I', choices=['R', 'L', 'A', 'P', 'I', 'S'], type=str, required=True, metavar='<order>', help='Order of the imaging planes in terms of the anatomical direction. Note that this direction specifies the origin rather than the target. Hence, if the imaging planes are ordered from inferior to superior, the order will be \'I\'.')
    __subparser['from-plane-images'].add_argument('--plane-orientation', dest='plane_orientation', nargs=2, default=['A', 'L'], choices=['R', 'L', 'A', 'P', 'I', 'S'], type=str, required=True, metavar='<orientation>', help='Orientation of the imaging planes in terms of the anatomical directions of the first (vertical) and the second (horizontal) axis. Note that an image\'s origin is in the upper left corner and that these directions specify the origins rather than the targets. Hence, if the axes go from anterior to posterior and left to right, their orientation will be \'A\' and \'L\', respectively.')
    __subparser['from-plane-images'].add_argument('--plane-rotation', dest='plane_rotation', default=0, type=float, required=False, metavar='<angle>', help='Angle of an in-plane rotation required to be added to fully align the image with its specified orientation. The angle is measured in degrees, positively in counter-clockwise direction and negatively in clockwise direction.')
    __subparser['from-plane-images'].add_argument('--plane-spacing', dest='plane_spacing', nargs=2, default=[1., 1.], type=float, required=True, metavar='<spacing>', help='Spacing of the imaging planes along the first (vertical) and the second (horizontal) axis.')
    __subparser['from-plane-images'].add_argument('--plane-height', dest='plane_height', default=1., type=float, required=True, metavar='<height>', help='Height of the imaging planes or, equivalently, the distance between imaging planes.')
    __subparser['from-plane-images'].add_argument('--as-mask', dest='as_mask', action='store_true', required=False, help='Indicates whether to create this volume image as a mask. In this case, the image intesities will be normalised and afterwards a threshold applied to produce a binary mask.')
    __subparser['from-plane-images'].add_argument('--output-file', dest='output_file', type=str, required=True, metavar='<path>', help='Path for the output file. If the file does already exist, it will not be overwritten and rather an error thrown.')
    
    # *** 'to-plane-images' ***
    
    __subparser['to-plane-images'].add_argument('--volume-image', dest='volume_image', type=str, required=True, metavar='<file name>', help='Volume image.')
    __subparser['to-plane-images'].add_argument('--plane-image-order', dest='image_order', default='I', choices=['R', 'L', 'A', 'P', 'I', 'S'], type=str, required=True, metavar='<order>', help='Order of the imaging planes in terms of the anatomical direction of the image stack. Note that this direction specifies the origin rather than the target. Hence, if the imaging planes are to be ordered from inferior to superior, the order is \'I\'.')
    __subparser['to-plane-images'].add_argument('--plane-orientation', dest='plane_orientation', nargs=2, default=['A', 'L'], choices=['R', 'L', 'A', 'P', 'I', 'S'], type=str, required=True, metavar='<orientation>', help='Orientation of the imaging planes in terms of the anatomical directions of the first (vertical) and the second (horizontal) axis. Note that an image\'s origin is in the upper left corner and that these directions specify the origins rather than the targets. Hence, if the axes go from anterior to posterior and left to right, their orientation will be \'A\' and \'L\', respectively.')
    __subparser['to-plane-images'].add_argument('--output-file', dest='output_file', type=str, required=True, metavar='<path>', help='Path for the output file. If the file does already exist, it will not be overwritten and rather an error thrown.')
    
    # *** 'resample' ***
    
    __subparser['resample'].add_argument('--volume-image', dest='volume_image', type=str, required=True, metavar='<file name>', help='Volume image.')
    __subparser['resample'].add_argument('--spacing', dest='spacing', nargs=3, default=[1., 1., 1.], type=float, required=True, metavar='<spacing>', help='Spacing of the volume image in the sagittal, coronal, and transverse anatomical direction.')
    __subparser['resample'].add_argument('--output-file', dest='output_file', type=str, required=True, metavar='<path>', help='Path for the output file. If the file does already exist, it will not be overwritten and rather an error thrown.')
    
    
    
    kwargs = vars(__parser.parse_args())
    
    # ********************************************************************************
    # Preprocess arguments

    __parser_subcommand = kwargs.pop('subcommand')

    if __parser_subcommand == 'from-plane-images':
        if abs(kwargs['plane_rotation']) > 45:
            warnings.warn('The specified in-plane rotation angle exceeds +/-45°. It is possible and recommended to choose the plane orientation differently which will in turn require a smaller in-plane rotation angle.', UserWarning)

        if abs(kwargs['plane_rotation']) == 45:
            warnings.warn('The specified in-plane rotation angle is +/-45°. This will make the orientation of the volume image being not well-defined. It is therefore suggested to slightly decrease the angle and make it e.g. +/-44.999999° instead.', UserWarning)

        kwargs['plane_rotation'] = math.radians(kwargs['plane_rotation'])

    if __parser_subcommand == 'to-plane-images':
        pass
    
    if __parser_subcommand == 'resample':
        pass

    # ********************************************************************************
    # Execute main function
    
    try:
        if __parser_subcommand == 'from-plane-images':
            volume_image_from_image_stack(**kwargs)
        elif __parser_subcommand == 'to-plane-images':
            volume_image_to_image_stack(**kwargs)
        elif __parser_subcommand == 'resample':
            volume_image_resample(**kwargs)
            
    except:
        print('', file=sys.stdout)
        print('', file=sys.stdout)
        print('An error occured. Operation could not be completed.', file=sys.stdout)
        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1)

    sys.exit(0)
