# ******************************************************************************
# This file contains functions to create and manipulate volume imagery. The
# basis for this functionality is the ANTsPy medical imaging library.
#
# This code shall be licensed under the GNU General Public License v3.0.
#
# (c) 2020- , Jan Mölter
# ******************************************************************************

__all__ = ['from_sagittal_image_stack'
           'from_coronal_image_stack'
           'from_transverse_image_stack'
           'from_image_stack'
           'to_sagittal_image_stack'
           'to_coronal_image_stack'
           'to_transverse_image_stack'
           'to_image_stack'
           'from_image_stack__'
           'to_image_stack__',
           'interpolate_image_stack',
           'load',
           'save',
           'mask_optimisation',
          ]


import os

import math

import ants
from . import antsX
import numpy

import scipy.spatial.transform

import scipy.ndimage.morphology
import scipy.ndimage.measurements

import skimage
import skimage.measure


def from_sagittal_image_stack(image_stack, image_rotation=0, image_spacing=(1.,1.), image_height=1.):
    """
    Converts a sagittal stack of 2-dimensional image rasters into a volumetric image
    in RAI orientation.
    
    Parameters
    ----------
    image_stack : list of (H,W) ndarray
        List of 2-dimensional sagittal image rasters, from right to left.
        Specifically, the orientation of these arrays has to be as shown in the
        schematic.
        
             [0,0]         [0,-1]
                  ┌───────┐
                  │   S   │
                  │ P ┼ A │
                  │   I   │
                  └───────┘
            [-1,0]         [-1,-1]    
    
    image_rotation : float or int
        Rotation angle of the image. This is the angle of rotation require to
        properly align the image with its specified orientation (Units: 1rad). Note
        that angles are measured positively in counter-clockwise direction and
        negatively in clockwise direction. Default is 0.
    image_spacing : (2,) tuple of float or int
        Spacing of the 2-dimensional image rasters, the width of the pixels along
        the first and second axis of the image rasters (Units: 1µm), respectively.
        Default is (1., 1.).
    image_height : float or int
        Height of the 2-dimensional image rasters (Units: 1µm). Default is 1..

    Returns
    -------
    _ : ants.core.ants_image.ANTsImage
        Volumetric image in RAI orientation.
    """
    
    
    __IMAGE = from_image_stack(image_stack, 'L', ('I', 'A'), image_rotation=image_rotation, image_spacing=image_spacing, image_height=image_height)
    
    return __IMAGE

def from_coronal_image_stack(image_stack, image_rotation=0, image_spacing=(1.,1.), image_height=1.):
    """
    Converts a coronal stack of 2-dimensional image rasters into a volumetric image
    in RAI orientation.
    
    Parameters
    ----------
    image_stack : list of (H,W) ndarray
        List of 2-dimensional coronal image rasters, from anterior to posterior.
        Specifically, the orientation of these arrays has to be as shown in the
        schematic.
        
             [0,0]         [0,-1]
                  ┌───────┐
                  │   S   │
                  │ L ┼ R │
                  │   I   │
                  └───────┘
            [-1,0]         [-1,-1]    
    
    image_rotation : float or int
        Rotation angle of the image. This is the angle of rotation require to
        properly align the image with its specified orientation (Units: 1rad). Note
        that angles are measured positively in counter-clockwise direction and
        negatively in clockwise direction. Default is 0.
    image_spacing : (2,) tuple of float or int
        Spacing of the 2-dimensional image rasters, the width of the pixels along
        the first and second axis of the image rasters (Units: 1µm), respectively.
        Default is (1., 1.).
    image_height : float or int
        Height of the 2-dimensional image rasters (Units: 1µm). Default is 1..

    Returns
    -------
    _ : ants.core.ants_image.ANTsImage
        Volumetric image in RAI orientation.
    """
    
    
    __IMAGE = from_image_stack(image_stack, 'P', ('I', 'R'), image_rotation=image_rotation, image_spacing=image_spacing, image_height=image_height)
    
    return __IMAGE

def from_transverse_image_stack(image_stack, image_rotation=0, image_spacing=(1.,1.), image_height=1.):
    """
    Converts a transverse stack of 2-dimensional image rasters into a volumetric
    image in RAI orientation.
    
    Parameters
    ----------
    image_stack : list of (H,W) ndarray
        List of 2-dimensional transverse image rasters, from inferior to superior.
        Specifically, the orientation of these arrays has to be as shown in the
        schematic.
        
             [0,0]         [0,-1]
                  ┌───────┐
                  │   A   │
                  │ L ┼ R │
                  │   P   │
                  └───────┘
            [-1,0]         [-1,-1]    
    
    image_rotation : float or int
        Rotation angle of the image. This is the angle of rotation require to
        properly align the image with its specified orientation (Units: 1rad). Note
        that angles are measured positively in counter-clockwise direction and
        negatively in clockwise direction. Default is 0.
    image_spacing : (2,) tuple of float or int
        Spacing of the 2-dimensional image rasters, the width of the pixels along
        the first and second axis of the image rasters (Units: 1µm), respectively.
        Default is (1., 1.).
    image_height : float or int
        Height of the 2-dimensional image rasters (Units: 1µm). Default is 1..

    Returns
    -------
    _ : ants.core.ants_image.ANTsImage
        Volumetric image in RAI orientation.
    """
    
    
    __IMAGE = from_image_stack(image_stack, 'S', ('P', 'R'), image_rotation=image_rotation, image_spacing=image_spacing, image_height=image_height)
    
    return __IMAGE

def from_image_stack(image_stack, stack_orientation, image_orientation, image_rotation=0, image_spacing=(1.,1.), image_height=1.):
    """
    Converts a stack of 2-dimensional image rasters into a volumetric image in RAI
    orientation.
    
    Parameters
    ----------
    image_stack : list of (H,W) ndarray
        List of 2-dimensional image rasters.
    stack_orientation : str
        Stack orientation in terms of anatomical directions. Possible values are 'R'
        or 'L' for a sagittal stack, 'A' or 'P' for a coronal stack, and 'I' or 'S'
        for a transverse stack. Note that these direction specify the target rather
        than the origin. Hence, for a transverse stack in order from inferior to
        superior the direction specifier is 'S'.
    image_orientation : (2,) tuple of str
        Image orientation in terms of anatomical directions of the two axes. Possible
        values are 'R', 'L', 'A', 'P', 'I' and 'S'. Similarly as with the stack
        orientation, these directions specify the targets rather than the origin.
        Note also that an image's vertical axis is the first and the horizontal axis
        the second. Hence, for a transverse image in standard orientation
        
             [0,0]         [0,-1]
                  ┌───────┐
                  │   A   │
                  │ L ┼ R │
                  │   P   │
                  └───────┘
            [-1,0]         [-1,-1]
            
        the image orientation is ('P', 'R').
            
        Similarly, for a sagittal image in standard orientation
        
             [0,0]         [0,-1]
                  ┌───────┐
                  │   S   │
                  │ P ┼ A │
                  │   I   │
                  └───────┘
            [-1,0]         [-1,-1]
            
        it is ('I', 'A') and for a coronal image in standard orientation
        
             [0,0]         [0,-1]
                  ┌───────┐
                  │   S   │
                  │ L ┼ R │
                  │   I   │
                  └───────┘
            [-1,0]         [-1,-1]
            
        it is ('I', 'R').
    image_rotation : float or int
        Rotation angle of the image. This is the angle of rotation require to
        properly align the image with its specified orientation (Units: 1rad). Note
        that angles are measured positively in counter-clockwise direction and
        negatively in clockwise direction. Default is 0.
    image_spacing : (2,) tuple of float or int
        Spacing of the 2-dimensional image rasters, the width of the pixels along
        the first and second axis of the image rasters (Units: 1µm), respectively.
        Default is (1., 1.).
    image_height : float or int
        Height of the 2-dimensional image rasters (Units: 1µm). Default is 1..

    Returns
    -------
    _ : ants.core.ants_image.ANTsImage
        Volumetric image in RAI orientation.
    """
    
    if not (type(image_stack) in [list, tuple] and all([type(_) is numpy.ndarray for _ in image_stack])):
        raise TypeError('`image_stack` is expected to be of type list with elements of type numpy.ndarray.')
        
    if not type(stack_orientation) is str:
        raise TypeError('`stack_orientation` is expected to be of type str.')
        
    if not (type(image_orientation) in [list, tuple] and len(image_orientation) == 2 and all([type(_) is str for _ in image_orientation])):
        raise TypeError('`image_orientation` is expected to be of type tuple with length 2 and with elements of type str.')

    if not type(image_rotation) in [int, float]:
        raise TypeError('`image_height` is expected to be either of type int or float.')

    if not (type(image_spacing) in [list, tuple] and len(image_spacing) == 2 and all([type(_) in [int, float] for _ in image_spacing])):
        raise TypeError('`image_spacing` is expected to be of type tuple with length 2 and with elements of type int or float.')

    if not type(image_height) in [int, float]:
        raise TypeError('`image_height` is expected to be either of type int or float.')

    if not all([len(_.shape) == 2 for _ in image_stack]):
        raise TypeError('`image_stack` is expected to contain only arrays of dimension 2.')
        
    if not len(set([_.shape for _ in image_stack])) == 1:
        raise ValueError('`image_stack` is expected to contain only elements with the same shape.')
    
    if not all([_ in ['R', 'L', 'A', 'P', 'I', 'S'] for _ in [stack_orientation] + list(image_orientation)]):
        raise ValueError('`stack_orientation` and `image_orientation` are expected to consist only of letters ''R'', ''L'', ''A'', ''P'', ''I'', and ''S''.')
    
    if abs(image_rotation) > math.pi / 4:
        pass
    
    
    def invert_orientation_specifier(orientation):
        __INVERSION_DICTIONARY = {'R': 'L', 'L': 'R', 'A': 'P', 'P': 'A', 'I': 'S', 'S': 'I'}
        return ''.join([__INVERSION_DICTIONARY[_] for _ in orientation])
    
    
    __IMAGE_ARRAY_ORIENTATION = invert_orientation_specifier(stack_orientation + ''.join(image_orientation))
    __IMAGE_ARRAY = numpy.stack(image_stack, axis=0)
    # `__IMAGE_ARRAY` is now in a natural orientation for the image stack.
    
    
    __SPACING = tuple([image_height] + list(image_spacing))
    __DIRECTION = antsX.direction_matrix(__IMAGE_ARRAY_ORIENTATION)
    
    if not image_rotation == 0:
        __ROTATION = scipy.spatial.transform.Rotation.from_rotvec(image_rotation * numpy.abs(__DIRECTION[:,0]))
        __DIRECTION = __ROTATION.apply(__DIRECTION.T).T
    
    __IMAGE = ants.from_numpy(__IMAGE_ARRAY, spacing=__SPACING, direction=__DIRECTION)
    # `__IMAGE` has been initialised from `__IMAGE_ARRAY` in its natural orientation.
    
    __IMAGE = ants.reorient_image2(__IMAGE, orientation='RAI')
    __IMAGE.set_origin((0,0,0))
    
    return __IMAGE


def to_sagittal_image_stack(image, indices=None, return_spacing=False):
    """
    Converts a volumetric image into a sagittal stack of 2-dimensional image rasters.
    
    Parameters
    ----------
    image : ants.core.ants_image.ANTsImage
        Volumetric image
    indices : list of int
        Indices of images to include in the stack.
    return_spacing : bool
        Include spacing information in return value. Default is False.

    Returns
    -------
    _ : list of (H,W) ndarray
        List of 2-dimensional sagittal image rasters, from right to left.
        Specifically, the orientation of these arrays has to be as shown in the
        schematic.
        
             [0,0]         [0,-1]
                  ┌───────┐
                  │   S   │
                  │ P ┼ A │
                  │   I   │
                  └───────┘
            [-1,0]         [-1,-1] 
    """
    
    
    return to_image_stack(image, 'L', ('I', 'A'), indices=indices, return_spacing=return_spacing)

def to_coronal_image_stack(image, indices=None, return_spacing=False):
    """
    Converts a volumetric image into a coronal stack of 2-dimensional image rasters.
    
    Parameters
    ----------
    image : ants.core.ants_image.ANTsImage
        Volumetric image
    indices : list of int
        Indices of images to include in the stack.
    return_spacing : bool
        Include spacing information in return value. Default is False.

    Returns
    -------
    _ : list of (H,W) ndarray
        List of 2-dimensional coronal image rasters, from anterior to posterior.
        Specifically, the orientation of these arrays has to be as shown in the
        schematic.
        
             [0,0]         [0,-1]
                  ┌───────┐
                  │   S   │
                  │ L ┼ R │
                  │   I   │
                  └───────┘
            [-1,0]         [-1,-1]  
    """
    
    
    return to_image_stack(image, 'P', ('I', 'R'), indices=indices, return_spacing=return_spacing)

def to_transverse_image_stack(image, indices=None, return_spacing=False):
    """
    Converts a volumetric image into a transverse stack of 2-dimensional image
    rasters.
    
    Parameters
    ----------
    image : ants.core.ants_image.ANTsImage
        Volumetric image
    indices : list of int
        Indices of images to include in the stack.
    return_spacing : bool
        Include spacing information in return value. Default is False.

    Returns
    -------
    _ : list of (H,W) ndarray
        List of 2-dimensional transverse image rasters, from inferior to superior.
        Specifically, the orientation of these arrays has to be as shown in the
        schematic.
        
             [0,0]         [0,-1]
                  ┌───────┐
                  │   A   │
                  │ L ┼ R │
                  │   P   │
                  └───────┘
            [-1,0]         [-1,-1]
    """
    
    
    return to_image_stack(image, 'S', ('P', 'R'), indices=indices, return_spacing=return_spacing)

def to_image_stack(image, stack_orientation, image_orientation, indices=None, return_spacing=False):
    """
    Converts a volumetric image raster into a stack of 2-dimensional image rasters.

    Parameters
    ----------
    image : ants.core.ants_image.ANTsImage
        Volumetric image
    stack_orientation : str
        Stack orientation in terms of anatomical directions. Possible values are 'R'
        or 'L' for a sagittal stack, 'A' or 'P' for a coronal stack, and 'I' or 'S'
        for a transverse stack. Note that these direction specify the target rather
        than the origin. Hence, for a transverse stack in order from inferior to
        superior the direction specifier is 'S'.
    image_orientation : (2,) tuple of str
        Image orientation in terms of anatomical directions of the two axes. Possible
        values are 'R', 'L', 'A', 'P', 'I' and 'S'. Similarly as with the stack
        orientation, these directions specify the targets rather than the origin.
        Note also that an image's vertical axis is the first and the horizontal axis
        the second. Hence, for a transverse image in standard orientation
        
             [0,0]         [0,-1]
                  ┌───────┐
                  │   A   │
                  │ L ┼ R │
                  │   P   │
                  └───────┘
            [-1,0]         [-1,-1]
            
        the image orientation is ('P', 'R').
            
        Similarly, for a sagittal image in standard orientation
        
             [0,0]         [0,-1]
                  ┌───────┐
                  │   S   │
                  │ P ┼ A │
                  │   I   │
                  └───────┘
            [-1,0]         [-1,-1]
            
        it is ('I', 'A') and for a coronal image in standard orientation
        
             [0,0]         [0,-1]
                  ┌───────┐
                  │   S   │
                  │ L ┼ R │
                  │   I   │
                  └───────┘
            [-1,0]         [-1,-1]
            
        it is ('I', 'R').
    indices : list of int
        Indices of images to include in the stack.
    return_spacing : bool
        Include spacing information in return value. Default is False.

    Returns
    -------
    _ : list of (H,W) ndarray
        List of 2-dimensional image rasters.
    """
    
    if not type(image) is ants.core.ants_image.ANTsImage:
        raise TypeError('`image` is expected to be of type ants.core.ants_image.ANTsImage.')
        
    if not type(stack_orientation) is str:
        raise TypeError('`stack_orientation` is expected to be of type str.')
        
    if not (type(image_orientation) in [list, tuple] and len(image_orientation) == 2 and all([type(_) is str for _ in image_orientation])):
        raise TypeError('`image_orientation` is expected to be of type tuple with length 2 and with elements of type str.')

    if not (indices is None or (type(indices) in [list, tuple] and all([type(_) is int for _ in indices]))):
        raise TypeError('`indices` is expected to be of type list and with elements of type int.')

    if not image.dimension == 3:
        raise ValueError('`image` is expected to be of dimension 3.')
    
    if not all([_ in ['R', 'L', 'A', 'P', 'I', 'S'] for _ in [stack_orientation] + list(image_orientation)]):
        raise ValueError('`stack_orientation` and `image_orientation` are expected to consist only of letters ''R'', ''L'', ''A'', ''P'', ''I'', and ''S''.')

        
        
    def invert_orientation_specifier(orientation):
        __INVERSION_DICTIONARY = {'R': 'L', 'L': 'R', 'A': 'P', 'P': 'A', 'I': 'S', 'S': 'I'}
        return ''.join([__INVERSION_DICTIONARY[_] for _ in orientation])
    
    
    __IMAGE_ARRAY_ORIENTATION = invert_orientation_specifier(stack_orientation + ''.join(image_orientation))
    
    __IMAGE = ants.reorient_image2(image, orientation=__IMAGE_ARRAY_ORIENTATION)
    __IMAGE_ARRAY = __IMAGE.view()
    # `__IMAGE_ARRAY` is now in a natural orientation to extract the image stack.
    
    if indices is not None:
        __IMAGE_STACK = [__IMAGE_ARRAY[_,:,:].copy() for _ in indices]
    else:
        __IMAGE_STACK = [__IMAGE_ARRAY[_,:,:].copy() for _ in range(__IMAGE_ARRAY.shape[0])]
    #__IMAGE_STACK = [numpy.squeeze(_) for _ in numpy.split(__IMAGE_ARRAY, __IMAGE_ARRAY.shape[0], axis=0)]
    
    if return_spacing:
        return __IMAGE_STACK, __IMAGE.spacing
    else:
        return __IMAGE_STACK


def normalise(image, range=(0,1), as_mask=False):
    """
    Normalises the image intensities of a volumetric image.
    
    Parameters
    ----------
    image : ants.core.ants_image.ANTsImage
        Volumetric image
    range : tuple or list of int or float
        Range of intensity values. Default is (0,1).
    as_mask : bool
        Threshold intensity values to produce a binary mask.
    
    Returns
    -------
    _ : ants.core.ants_image.ANTsImage
        Normalised volumetric image.
    """

    if as_mask:
        range = (0,1)

    __IMAGE = image.clone(pixeltype='float')

    __IMAGE[:,:,:] -= __IMAGE[:,:,:].min()
    __IMAGE[:,:,:] /= __IMAGE[:,:,:].max()

    __IMAGE[:,:,:] *= range[1] - range[0]
    __IMAGE[:,:,:] += range[0]

    if as_mask:
        __IMAGE = ants.threshold_image(__IMAGE, low_thresh=0.5, binary=True).astype('uint8')

    return __IMAGE


def interpolate_image_stack(image_stack, stack_spacing, target_spacing):
    """
    Linearly interpolates a non-uniformly spaced stack of 2-dimensional image
    rasters and yields a uniformly spaced stack.

    Parameters
    ----------
    image_stack : list of (H,W) ndarray
        List of 2-dimensional image rasters.
    stack_spacing : list or tuple of float or int, or float or int
        Spacing between the 2-dimensional image rasters in the stach (Units: 1µm).
    target_spacing : float or int
        Spacing between 2-dimensional image rasters in the interpolated stack.
        (Units: 1µm).

    Returns
    -------
    _ : list of (H,W) ndarray
        Interpolated image stack
    """
    
    if not (type(image_stack) in [list, tuple] and all([type(_) is numpy.ndarray for _ in image_stack])):
        raise TypeError('`image_stack` is expected to be either of type list or tuple with elements of type numpy.ndarray.')
    
    if not type(stack_spacing) in [list, tuple]:
        if type(stack_spacing) in [int, float]:
            stack_spacing = tuple([stack_spacing])
        else:
            raise ValueError('`stack_spacing` is expected to be either of type list or tuple with elements of type int or float.')
    
    if len(stack_spacing) == 1:
        stack_spacing = tuple([stack_spacing[0]] * (len(image_stack) - 1))
    
    if not len(image_stack) - 1 == len(stack_spacing):
        raise ValueError('Length of `stack_spacing` is expected to be 1 less than the length of `image_stack`.')
    
    if not (all([_ > 0 for _ in stack_spacing]) and target_spacing > 0):
        raise ValueError('`stack_spacing` and `target_spacing` are expected to be greater than 0.')
    
    if not all([_ % target_spacing == 0 for _ in stack_spacing]):
        raise ValueError('`target_spacing` is expected to be a divisor of (all) `stack_spacing`.')

    
    if not len(set([_.shape for _ in image_stack])) == 1:
        raise ValueError('`image_stack` is expected to contain only elements with the same shape.')
    
    
    def linear_interpolator(range, t):
        return (1-t) * range[0] + t * range[1]

    __interpolated_image_stack = [1. * image_stack[0]]

    for i in range(len(stack_spacing)):
        for t in [_ / stack_spacing[i] for _ in range(1, int(stack_spacing[i] / target_spacing)+1)]:
            __interpolated_image_stack.append(linear_interpolator((image_stack[i], image_stack[i+1]), t))
            
            
    return __interpolated_image_stack

def load(path, pixeltype='float'):
    """
    Loads a 3-dimensional image raster.

    Parameters
    ----------
    path : str
        Image path.
    pixeltype : str
        Pixeltype of the image raster Possible values are 'unsigned char', 'unsigned
        int', 'float', or 'double'. Default is 'float'.

    Returns
    -------
    _ : ants.core.ants_image.ANTsImage
        3-dimensional image raster.
    """
    
    if not type(path) is str:
        raise TypeError('`path` is expected to be of type str.')
        
    if not os.path.isfile(path):
        raise FileNotFoundError('The file {} does not exist.'.format(path))
    
    return ants.image_read(path, pixeltype=pixeltype)

def save(volume_image, output_path, override=False):
    """
    Saves a 3-dimensional image raster to file.

    Parameters
    ----------
    volume_image: ants.core.ants_image.ANTsImage
        3-dimensional image raster.
    output_path : str
        Image output path.
    override : bool
        Override existing files. Default is False.

    Returns
    -------
    None
    """
    
    if not type(volume_image) is ants.core.ants_image.ANTsImage:
        raise TypeError('`volume_image` is expected to be of type ants.core.ants_image.ANTsImage.')
    
    if not type(output_path) is str:
        raise TypeError('`output_path` is expected to be of type str.')
        
    if os.path.isfile(output_path) and not override:
        raise FileExistsError('The file {} does already exist and `override` is set to False.'.format(path))
        
    volume_image.to_file(output_path)

def anatomically_annotate(image):

    import base64
    import zlib
    
    LETTERS = {
        'R' : b'eNrtmj9v4zAMxT+Qh+Ra984ZOlCSE7j/zho6eM6gIUOGDsLdp784aYEWV4uySEoe8oCsgeCf+UQ+GuCqqwJ6NiuSGgO+0/BqKIcwsOLSRrtBpZ3C9ytWve0K8PhG9a4gj09am7I83uWHwjwuqmxpHmf9LM7jrH1xHmfzKM9jtFRVnkf0+ynK46S/5XmctAAeJ/1ZAI9V5Sg8bKsA++0fjEVrhMLDR5a51tAEmW5sOo8m3nh1+P0+EHjMupAfDQkImce71j2hUDl4XJgQ6pSJx0nHPv0u4+IB2+lnYXPxAJj2jDofD5jkUeXjAdM8XD4e0551uwQeGevDLKE+lF+CX017d5ONx2tP6POYeLQ9oUaZeLQvhtR3c/CwgecQYZl0Hkq5cN8eMwfN4KG+9rqu0+Bf8J69W8D8cVzAPNjoBcyDt3GvliiPWLuR5FFBZIYlyQOi3UaOR/wZxHhswM1oSGR43IGdc/9I8Kjb4nm7G+Y2Rdw8fikLCeLjkb6D4eFRwaCBIB4ennQGPh57yiEmeWw+9TEaDjg3aAV4/NfT7oxj82pKv9shz+OGm8f3DcFzHz7HIMzj49EhOaoX53GZwz07k4T5wyJ1PcjzCOcBiXWSNA8+hd/PWsnzGJHU4RnQy/MYzYvXM1Lncxd+P30nz2N8hpZl/iHmJR3iny4DD7xWGy/PA69Vn4HHWKtcdwkpv9oahpyCnl+99fS9NTlPVMi9+iMDj3Cue/6XNgMPAO2RGVmeB5IvR+0dWPJdZcmf23Dk7U9UD2fJ2xHfwvMblv3HlujhTPuPA+m7CqZ9lArPith3WFz7KEvZP7DtB3W4Vtd59oPtb5Ncqnz7cxXOf+o8+3N4MIk7Icb9OTYbNTl44LPRkIMH6ltT+TcrD9S3jnDVVSHd3/8D53jAOw==',
        'L' : b'eNrt0bEJwCAUQMGBLGxsLQxJn0zh/htkgQiBLzbem+DBtSb967nzd2XhxHUOJtLCiaMNJnLnwYMHDx48ePDgwYMHDx48ePDgwYMHDx48ePDgwYMHDx48ePDgwYMHDx48Qh6x+iSPUGWSR6jEgwcPHjx4bOKhvav1BSpUZAA=',
        'A' : b'eNrtWrFy2zAM/SANdFsnkYcMlKCmqquYHDx4zsDBgwcPTPP1aate79waEAES0vXO7y5b7HsmH/AAENbecEMJNMFg+Dgfi5VDWfi5OAw4B2PeZiKxBYJEFeaRBESChKmbhXX5C5tZziI6msVqBg79BIcfNxLVSXyFKRLmpE6C1uUI65fV5YiD8kmcXQKJzdK6HDGokoA0EvequgxpJEzs9HTpTSo+LazLEWFpXaom8C/AIKHl68m6HPGqokvL4mA2Kl52cjwWawUOHcahgvkSOKbLCve18r4OHrv7zmFnUZf2dbSW6G2LnsXLTLr8GQQe1UtpXQLR/+E5pOiFPNH3Htwcvo7p8nd+bvGcXtDX0fPeT/UkBX396CbyInpfBX0d0+Xdn//wtVP2dVyXcbpnL+XrYKdDkPDZV11dXuTEHaj6OpoHLn5ja1V9HfuND6l16CY/dX6GxLkIqh1zzibRYh76z10f0LOog5YuT4y5VtDS5YpRi1ZKurz2vR3uZfusYgJYP26NncVDli4DXtvxZgi9nIRntpxBoxl5cczZ/qDg6ztuTU/M2D5ILZTv0XgvUsXSutwLPmO+y3SJ1Uw19SE0f8uSFuoHd7J5hihEnhNqO06ul/g6COON6EX4vt56ab8bC/p6zZzRaAwZDwoc2CEygNEA602i1eFAJhmGLnPRL6zL6Vx3ib0Whyu1OopvoEYiedegVeRgTCoJq0ki1dfXTpNE5RfWZXq62IIuiZRdg0abQ0qgNlGdRIKvr50+i+NUkYvXaMF3jWX8BXG62BbspXYg8/WiMw9CX2SgNkV7mM7JfB3dMautAG2U+Prgyu4zEfk3SvKl7B0DBDuEuC6l741Hxw1UhTkDVZ+8MXuojHkg3pddn3fgs+pKPqvG34iu7xASO2Y5O0Q1K+ZwXWY9KRExt+LoMu+NL3J2CM9aM/tnSN4hJHSZ+Z5E1Up/7RD2Qu/N9LJDYo6tbS6IGePFCgqxY1bgHd67tB1CXJcl9mUIL7tPurcib77BLbFDeMP/jcfHd2yiAZc=',
        'P' : b'eNrt2b1ugzAQB/AHyuAoRZUzZDhjRFBFaw8Z8gQeMzD49QtNo0hVcIE7fwz3lzKCHH7Y+M4AHM6C9FpsylGD72vrzgoa9CA0CHzcrQWLGYU3giTyA1xyj1d53/o8SDyez6PN6/HITWX2uMc2mT3u83f9u0HtMWWvsntMye8xpcrv8fOC5vcQYleAx5hrAR6rnkUsj1XPYt7DNgpe/S5dDcPngrFLAg8fvq5ptf1vGB3aY8EfabUPDuKA91hydRN02fnYHo8MhmDpxHhMUaF3Q9oUHlOswa/fWI/guntI5DHmYrALJ95jvMd8YZLMA2BvkBOEwGP+Hkk9oAQPX4LHVwEeqoT5odzsIN7SeczO0bEoS+VxDXw/+kQenQnsK3waj+DeRkISjz68v4q+3522E86E9949RPbQtZWGqjxe5tE86w/rutovqt8qgrUf366w+G9xyqI0Wn3u1/TSYnms6RHE8hgK6F9VK3t5MTykLqCfuM/f35UbDmWoPSq9fgzUHsOWnj+th7tkPo8Sot5+RkfkUZ0xZ3N4D9nX0AMu2zx+z0lbBR0QZNbj+KenGvPQmKI+x4eiPo/nIRMOgj3Ygz3Ygz3Ygz3Ygz3Ygz3Ygz3Ygz3Yg8OZcjp9A+i2Dc0=',
        'I' : b'eNrt0aENACAMAMGRwCIQJeBh/2WYoI6g7if45CKkF40oSefjxdnJRP04sWYy0Xjw4MGDBw8ePHjw4MGDBw8ePHjw4MGDBw8ePHjw4MGDBw8ePHjw4MGDBw8ePHjw4MGDBw/pUb1ftCancg==',
        'S' : b'eNrtWj1v2zAQ/UEapLZKIw8ZjqQiM4pacfCgOQMHDRk8sP73lZMiQIrw43hHBSj6gCwBbD8e7/PxAP4jG1IYOypo5vod2kk6O4Db4ffteVZ1BGD7YgzM8iP6+2/oRsHPQEjo5hqHSlhOCr1Wrs5BJQzbRfxUdTY6YLHHOtckdPd0MzgihyueiBwaBg6bc9jP57DhJp/EmYvDhiWTwxMjh81Ds3L6xMphqy85aXRSvCTqCp+77rk5bLhgSSjLTwIbqwKidzxKOPUCzOufW1Pq7IUxPivh9Ee1blCG0y/6AId2CPqSDZc0Hr98jsRaH7yXA8IW0mvXZ2K+XxGm8NXPpMQnAn3QAUHCd5YvaR9fArZA5AmiZwX8s0m/Ds83JPuV3xbpvmk839Empzt/jCQnzt57H4bq24iDALk/0b5zVPT4qMDgzqHATQpWLcHq11qjyfFR1+cdR/BA3oN+LxKPoRog3D4kRHgGreQu9oj3upVbRGkWaT3eeQBTkoVN7fsPshwRieu5nS5zOc/4GQisYrdFlk7TjYJVQsufSSv3JNliBGoCHFPs6Jk2E3Yji87qaio6hrgZGOZjBh4Lh15xQ/UPYToGHi01h6jYvFtWv4r3jRgcyDqvHhl4fGXIGww8Fo789aCIuYPpnWiQJB4NV1kR1zep7PrG+zY1Zmp9F+CFEu6MrnOVhQKYjkg/uUAhCFiT3w4PRWeofoudlFrTQGkIE++TOwvlYWzEXxfYBSb4pvh9rwE/2IvsJ3aE3k9wEWLNvQQ7qevbOeMsF9dGdS+M1Qrc3/6FjfDAXDtl91IHtC28+lM0WUjg0OtfMOafx8xcpdh/HkO4S+wQYef85O2NcWyemQj+FdASNYqEvy+NaxkBLfEWlTctJV+FtMQTgkU7U8I91EOm7834tcC0Pi9UB5vEkT801yfVcxXsCYwmcqh/kTX/q95u8u8C8bwX02puQ74hImdIn9PbWL/Yfaxd6mNU13DJmmeabgXr8W2fAU5a2jah57Yccb6npNarMiS+4UrQWsIW2PZI8Ghn1B2saea+k5x5VAAvh0x9YFCfz+GlDqhP53DV/TueXU2a1i0Ug28w7BIfiVrmLc8ecf+Q7xsN40NdprYLJ2AGUtttH6CMRiSESdHs2sePdwd5RTvh/ugX76vD9r/rLstuux//JO7ufgM3kfCj',
    }
    
    def text_transform_array(_, shape=None):
        
        if type(_) is numpy.ndarray:
            
            assert _.dtype == 'uint8'
            
            __BUFFER = base64.b64encode(_.reshape(1,-1))
            __BUFFER = zlib.compress(__BUFFER, level=9)
            __BUFFER = base64.b64encode(__BUFFER)
            
            return __BUFFER
        
        if type(_) is bytes:
            
            __BUFFER = base64.b64decode(_)
            __BUFFER = zlib.decompress(__BUFFER)
            __BUFFER = base64.b64decode(__BUFFER)
            
            if shape is not None:
                return numpy.frombuffer(__BUFFER, dtype='uint8').reshape(*shape)
            else:
                return numpy.frombuffer(__BUFFER, dtype='uint8')
    
    for _ in LETTERS.keys():
        LETTERS[_] = text_transform_array(LETTERS[_], shape=(100,100))
        LETTERS[_] = LETTERS[_].astype('float')
        LETTERS[_] = (LETTERS[_] - LETTERS[_].min()) / (LETTERS[_].max() - LETTERS[_].min())
    
        
    
    __IMAGE = image.clone()
    __IMAGE_MAX = __IMAGE.max()
    
    __IMAGE_STACK, __IMAGE_STACK_SPACING = to_transverse_image_stack(__IMAGE, return_spacing=True)
    
    for i in range(len(__IMAGE_STACK)):
        __I = __IMAGE_STACK[i]
    
        __C = LETTERS['L']
        if not (100, 100) < __I.shape:
            __C = __C[::100 // (min(__I.shape)//2),::100 // (min(__I.shape)//2)]
        __I[__I.shape[0]//2-__C.shape[0]//2:__I.shape[0]//2-__C.shape[0]//2+__C.shape[0],0:__C.shape[1]] += __IMAGE_MAX * __C
        
        __C = LETTERS['A']
        if not (100, 100) < __I.shape:
            __C = __C[::100 // (min(__I.shape)//2),::100 // (min(__I.shape)//2)]
        __I[0:__C.shape[0],__I.shape[1]//2-__C.shape[1]//2:__I.shape[1]//2-__C.shape[1]//2+__C.shape[1]] += __IMAGE_MAX * __C
    
    __IMAGE = from_transverse_image_stack(__IMAGE_STACK, tuple(__IMAGE_STACK_SPACING[1:]), __IMAGE_STACK_SPACING[0])
    
    
    __IMAGE_STACK, __IMAGE_STACK_SPACING = to_coronal_image_stack(__IMAGE, return_spacing=True)
    
    for i in range(len(__IMAGE_STACK)):
        __I = __IMAGE_STACK[i]
        
        __C = LETTERS['S']
        if not (100, 100) < __I.shape:
            __C = __C[::100 // (min(__I.shape)//2),::100 // (min(__I.shape)//2)]
        __I[0:__C.shape[0],__I.shape[1]//2-__C.shape[1]//2:__I.shape[1]//2-__C.shape[1]//2+__C.shape[1]] += __IMAGE_MAX * __C
    
    __IMAGE = from_coronal_image_stack(__IMAGE_STACK, tuple(__IMAGE_STACK_SPACING[1:]), __IMAGE_STACK_SPACING[0])
    
    
    __IMAGE[__IMAGE[:,:,:] > __IMAGE_MAX] = __IMAGE_MAX
    
    return __IMAGE


def ellipsoid(radius, dtype=numpy.uint8):
    """
    Generates an ellipsoidal structuring element. This extends the collection of
    structure elements defined in the scikit-image library.
    
    Parameters
    ----------
    radius : tuple of int
        Radii of the ellipsoidal structuring element.

    Returns
    -------
    selem : ndarray
        The structuring element where elements of the neighborhood are 1 and 0
        otherwise.
    """
    
    if not (type(radius) in [tuple, list] and all([type(_) is int for _ in radius])):
        raise TypeError('`radius` is expected to be of either type tuple or list with elements of type int.')
    
    if not all([_ > 0 for _ in radius]):
        raise ValueError('All elements of `radius` are expected to be positive.')
    
    radius = numpy.array(radius)
    
    n = 2 * radius + 1
    S = numpy.moveaxis(numpy.indices(tuple(n)), 0, -1)
    
    S = numpy.sum(((S - n // 2) / radius) ** 2, axis=len(n))
    S = numpy.where(S > 1, 0, 1).astype(dtype)
    
    return S

def mask_optimisation(mask, dilation_erosion_radius=None, min_connected_component_size=None):
    """
    Optimises mask images by dilation-erosion and the discrimination of small
    connected components.

    Parameters
    ----------
    mask: ants.core.ants_image.ANTsImage
        3-dimensional mask image.
    dilation_erosion_radius : float or int
        Dilation-erosion radius (Units: 1µm). Default is None.
    min_connected_component_size : float or int
        Minimal volume of a connected component (Units: 1µm³). Default is None.

    Returns
    -------
    _ : ants.core.ants_image.ANTsImage
        Optimised mask.
    """

    if not type(mask) is ants.core.ants_image.ANTsImage:
        raise TypeError('`mask` is expected to be of type ants.core.ants_image.ANTsImage.')

    if not (dilation_erosion_radius is None or type(dilation_erosion_radius) in [int, float]):
        raise TypeError('`dilation_erosion_radius` is expected to be either of type int or float.')

    if not (min_connected_component_size is None or type(min_connected_component_size) in [int, float]):
        raise TypeError('`min_connected_component_size` is expected to be either of type int or float.')

    if not mask.dimension in [2, 3]:
        raise ValueError('`mask` is expected to be of dimension 2 or 3.')
    
    __mask = mask.clone()

    assert set(__mask.unique()) == {0, 1}, '`mask` is not binary.'


    # Step 1: Dilation-Erosion

    DE_structure = ellipsoid([int(_) for _ in numpy.ceil(numpy.full(__mask.dimension, dilation_erosion_radius) / numpy.array(__mask.spacing))])

    __mask[:,:,:] = scipy.ndimage.morphology.binary_dilation(__mask[:,:,:], structure=DE_structure, border_value=0).astype(__mask.dtype)
    __mask[:,:,:] = scipy.ndimage.morphology.binary_erosion(__mask[:,:,:], structure=DE_structure, border_value=1).astype(__mask.dtype)
    
    
    # Step 2: Connected-Component discrimination
    
    __labeled_mask, N_labels = scipy.ndimage.measurements.label(__mask[:,:,:])

    print(__mask.spacing)
    print('*' * 80)
    for _ in range(1, N_labels+1):
        print(numpy.sum(__labeled_mask == _) * math.prod(__mask.spacing))
        if numpy.sum(__labeled_mask == _) * math.prod(__mask.spacing) < min_connected_component_size:
            #print('Volume:', numpy.sum(__labeled_mask == _) * math.prod(__mask.spacing), '(', numpy.sum(__labeled_mask == _), ')', min_connected_component_size)
            __mask[__labeled_mask == _] = 0


    return __mask.astype('uint8')


def find_mask_contours(mask, orientation='transverse', segment_length=0., use_voxels=False, include_holes=True):
    """
    Finds contours of the connected components of a mask image.
    
    Parameters
    ----------
    mask : ants.core.ants_image.ANTsImage
        Mask image.
    orientation : str
        Orientation of the contour stack, that can be either 'sagittal', 'coronal',
        or 'transverse'. Default is 'transverse'.
    segment_length : float
        Intended length of the line segments along the contour. Default is 0.
    use_voxels : bool
        Interpret the values of the segment length in units of voxels. Default is
        False.
    include_holes : bool
        Include contours of holes in connected components. Default is True.
    
    Returns
    -------
    _ : list of list of (n,2) ndarray
        Contours of the boundaries of the connected components.
    """
    
    if not type(mask) is ants.core.ants_image.ANTsImage:
        raise TypeError('`mask` is expected to be of type ants.core.ants_image.ANTsImage.')

    if not type(orientation) is str:
        raise TypeError('`orientation` is expected to be of type str.')

    if not type(segment_length) in [int, float]:
        raise TypeError('`segment_length` is expected to either be of type float or int.')
    
    if not mask.dimension in [2, 3]:
        raise ValueError('`mask` is expected to be of dimension 2 or 3.')

    if not orientation in ['sagittal', 'coronal', 'transverse']:
        raise ValueError('`orientation` is expected to be either ''sagittal'', ''coronal'', or ''transverse''.')

    if not segment_length >= 0:
        raise ValueError('`segment_length` is expected to be non-negative.')
    

    

    if orientation == 'sagittal':
        __image_stack, __image_stack_spacing = to_sagittal_image_stack(mask, return_spacing=True)
    elif orientation == 'coronal':
        __image_stack, __image_stack_spacing = to_coronal_image_stack(mask, return_spacing=True)
    elif orientation == 'transverse':
        __image_stack, __image_stack_spacing = to_transverse_image_stack(mask, return_spacing=True)


    contours = [None] * len(__image_stack)

    def component_contour(regionprop, inc_holes=include_holes):
    
        if inc_holes:
            image = regionprop.image
        else:
            image = regionprop.filled_image

        embedded_image = numpy.zeros(tuple(numpy.array(image.shape) + 2), dtype='uint8')
        embedded_image[1:image.shape[0]+1,1:image.shape[1]+1] = image

        contour = skimage.measure.find_contours(embedded_image, level=0.5)

        i, j, _, _ = regionprop.bbox

        contour = [_ - 1 + numpy.array([i,j]) for _ in contour]

        return contour

    for i in range(len(__image_stack)):

        __contours = []

        for regionprop in skimage.measure.regionprops(skimage.measure.label(__image_stack[i])):
            __contours += component_contour(regionprop)

        if not (segment_length is None or segment_length == 0):

            w = numpy.array(__image_stack_spacing[1:])
            if use_voxels:
                w[:] = 1.
            norm = lambda x : math.sqrt(numpy.sum((w * x)**2))

            for s in range(len(__contours)):

                n = 0
                while n < len(__contours[s]):
                    __ix = []
                    for dn in range(1, len(__contours[s])):
                        if n + dn == len(__contours[s]) or norm(__contours[s][n,:] - __contours[s][n + dn,:]) > segment_length:
                            __ix = list(range(n + 1, n + dn - 1))
                            break

                    if len(__ix) > 0:
                        __contours[s] = numpy.delete(__contours[s], __ix, axis=0)

                    n += 1

                if not numpy.all(__contours[s][0,:] == __contours[s][-1,:]):
                    contours[s] = numpy.append(__contours[s], __contours[s][[0],:], axis=0)

        contours[i] = __contours


    return contours
