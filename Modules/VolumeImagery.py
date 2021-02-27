# ******************************************************************************
# This file contains functions to create and manipulate volume imagery. The
# basis for this functionality is the ANTsPy medical imaging library.
#
# This code shall be licensed under the GNU General Public License v3.0.
#
# (c) 2020- , Jan Mölter
# ******************************************************************************

__all__ = ['from_image_stack',
           'to_image_stack',
           'interpolate_image_stack',
           'load',
           'save',
           'mask_optimisation',
          ]


import os

import ants
import numpy

import skimage


def from_image_stack(image_stack, image_spacing=(1.,1.), image_height=1., axis=2, spacing=None, normalise_values=False, binary_mask=False):
    """
    Converts a stack of 2-dimensional image rasters into a single 3-dimensional
	image raster.
    
    Parameters
    ----------
    image_stack : list of (W,H) ndarray
        List of 2-dimensional image rasters. If axis is set to 0 the order is from
        left to right, if axis is set to 1 it is from anterior to posterior, and if
        axis is set to 2 it is from inferior to superior.
    image_spacing : (2,) tuple of float or int
        Spacing of the 2-dimensional image rasters, the width of the pixels in the
        image rasters (Units: 1µm). Default is (1., 1.).
    image_height : float or int
        Height of the 2-dimensional image rasters (Units: 1µm). Default is 1..
    axis : int
        Stack axis. For vertical imaging, this is 2. Default is 2.
    image_rotation: int
        Rotation of the 2-dimensional image rasters in 90°-rotations. Default is 1.
    spacing : (3,) tuple of float or int
        Spacing of the 3-dimensional image raster (Units: 1µm). Default is None.
    normalise_values : bool
        Map pixel values to the range [0, 1]. Default is False.
    binary_mask : bool
        Indicates binary map images. Default is False.

    Returns
    -------
    _ : ants.core.ants_image.ANTsImage
        3-dimensional image raster.
    """
    

    if not (type(image_stack) in [list, tuple] and all([type(_) is numpy.ndarray for _ in image_stack])):
        raise TypeError('`image_stack` is expected to be of type list with elements of type numpy.ndarray.')

    if not (type(image_spacing) in [list, tuple] and len(image_spacing) == 2 and all([type(_) in [int, float] for _ in image_spacing])):
        raise TypeError('`image_spacing` is expected to be of type tuple with length 2 and with elements of type int or float.')

    if not type(image_height) in [int, float]:
        raise TypeError('`image_height` is expected to be either of type int or float.')

    if not (type(axis) is int and axis in [0, 1, 2]):
        raise TypeError('`axis` is expected to be of type int.')

    #if not type(image_rotation) is int:
    #    raise TypeError('`image_rotation` is expected to be of type int.')

    if not (spacing is None or (type(spacing) in [list, tuple] and len(spacing) == 3 and all([type(_) in [int, float] for _ in spacing]))):
        raise TypeError('`spacing` is expected to be of type tuple with length 3 and with elements of type int or float.')
    

    if not len(set([_.shape for _ in image_stack])) == 1:
        raise ValueError('`image_stack` is expected to contain only elements with the same shape.')
    
    #__volume_image_data = numpy.vstack([numpy.expand_dims(_, axis=0) for _ in image_stack])
    #
    ##__volume_image_data = numpy.flip(__volume_image_data, axis=0)
    ##if not image_rotation == 0:
    ##    __volume_image_data = numpy.rot90(__volume_image_data, k=image_rotation, axes=(1,2))
    #__volume_image_data = numpy.moveaxis(__volume_image_data, 0, axis)
    __volume_image_data = numpy.stack(image_stack, axis=axis)
    
    __volume_image_data = __volume_image_data.astype('float32')


    if normalise_values or binary_mask:
        __volume_image_data -= __volume_image_data.min()
        __volume_image_data /= __volume_image_data.max()
    
    #
    #
    #__nrrd_header = dict()
    #__nrrd_header['encoding'] = 'raw'
    #__nrrd_header['space dimension'] = 3
    #__nrrd_header['space directions'] = numpy.diag([pixel_width, pixel_width, plane_distance])
    #__nrrd_header['space units'] = ['microns'] * 3
    #
    #
    #nrrd.write(output_file, __volume_image, index_order='C', header=__nrrd_header)
    #
    
    __spacing = list(image_spacing)
    __spacing.insert(axis, image_height)
    __spacing = tuple(__spacing)

    __volume_image = ants.from_numpy(__volume_image_data, spacing=__spacing)
    if spacing is not None and __volume_image.spacing != spacing:
        __volume_image = ants.resample_image(__volume_image, spacing, use_voxels=False, interp_type=0)
    
    if binary_mask:
        __volume_image = ants.threshold_image(__volume_image, low_thresh=0.5, binary=True)
        __volume_image = __volume_image.astype('uint8')

            
    return __volume_image

def to_image_stack(volume_image, axis=2):
    """
    Converts a 3-dimensional image raster into a stack of 2-dimensional image
	rasters.

    Parameters
    ----------
    volume_image : ants.core.ants_image.ANTsImage
        3-dimensional volume image.
    axis : int
        Axis along which to list the image planes. If axis is set to 0 the order is
        from left to right, if axis is set to 1 it is from anterior to posterior, and
        if axis is set to 2 it is from inferior to superior. Default is 2.

    Returns
    -------
    _ : list of (W,H) ndarray
        List of image planes.
    """

    def ndarray_slice(dimensions, axis, index):
        _slice = [slice(None)] * dimensions
    
        if axis < dimensions:
            _slice[axis] = index
        
        return tuple(_slice)
    
    
    if not type(volume_image) is ants.core.ants_image.ANTsImage:
        raise TypeError('`volume_image` is expected to be of type ants.core.ants_image.ANTsImage.')
    
    if not type(axis) is int:
        raise TypeError('`axis` is expected to be of type int.')

    if not volume_image.dimension in [3]:
        raise ValueError('`volume_image` is expected to be of dimension 3.')
    
    if not axis in [0, 1, 2]:
        raise ValueError('`axis` is expected to be 0, 1, or 2.')
    
    #__image_stack = [volume_image[ndarray_slice(3, axis, i)] for i in range(volume_image.shape[axis])]
    __image_stack = [numpy.squeeze(_) for _ in numpy.split(volume_image.numpy(), volume_image.shape[axis], axis=axis)]

    return __image_stack

def interpolate_image_stack(image_stack, stack_spacing, target_spacing):
    """
    Linearly interpolates a non-uniformly spaced stack of 2-dimensional image
    rasters and yields a uniformly spaced stack.

    Parameters
    ----------
    image_stack : list of (W,H) ndarray
        List of 2-dimensional image rasters.
    stack_spacing : list or tuple of float or int, or float or int
        Spacing between the 2-dimensional image rasters in the stach (Units: 1µm).
    target_spacing : float or int
        Spacing between 2-dimensional image rasters in the interpolated stack.
        (Units: 1µm).

    Returns
    -------
    _ : list of (W,H) ndarray
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


def mask_optimisation(mask, axis=None, dilation_erosion_radius=None, min_connected_component_size=None):
    """
    Optimises mask images by dilation-erosion and the discrimination of small
    connected components.

    Parameters
    ----------
    mask: ants.core.ants_image.ANTsImage
        3-dimensional mask image.
    axis : int
        Axis along which to perform the dilation-erosion. Default is None.
    dilation_erosion_radius : float or int
        Dilation-erosion radius (Units: 1µm). Default is None.
    min_connected_component_size : float or int
        Minimal size of a connected component (Units: 1µm²). Default is None.

    Returns
    -------
    _ : ants.core.ants_image.ANTsImage
        Optimised mask.
    """

    def ndarray_slice(dimensions, axis, index):
        _slice = [slice(None)] * dimensions
    
        if axis < dimensions:
            _slice[axis] = index
        
        return tuple(_slice)
    
    def drop(list, index):
        list.pop(index)
        
        return list


    if not type(mask) is ants.core.ants_image.ANTsImage:
        raise TypeError('`mask` is expected to be of type ants.core.ants_image.ANTsImage.')

    if not (dilation_erosion_radius is None or type(dilation_erosion_radius) in [int, float]):
        raise TypeError('`dilation_erosion_radius` is expected to be either of type int or float.')

    if not (min_connected_component_size is None or type(min_connected_component_size) in [int, float]):
        raise TypeError('`min_connected_component_size` is expected to be either of type int or float.')

    if not mask.dimension in [2, 3]:
        raise ValueError('`mask` is expected to be of dimension 2 or 3.')
    
    
    __mask = mask.clone()
    __mask = __mask.astype('float32')
    
    if __mask.dimension == 3:
        if axis is None:
            raise ValueError('Dilations-erosion cannot be performed with axis set to None.')
        
        for i in range(__mask.shape[axis]):
            
            __slice = ndarray_slice(3, axis, i)
            
            __mask_slice = ants.from_numpy(__mask[__slice], spacing=drop(list(mask.spacing), axis))
            __mask_slice = mask_optimisation(__mask_slice, dilation_erosion_radius=dilation_erosion_radius, min_connected_component_size=min_connected_component_size)
            
            __mask[__slice] = __mask_slice.numpy()
        
        
    elif __mask.dimension == 2:
        
        if dilation_erosion_radius is not None:
            if not len(set(__mask.spacing)) == 1:
                raise ValueError('Dilation-erosion cannot be performed with non-uniform spacing.')
            
            __radius = dilation_erosion_radius / __mask.spacing[0]
            
            __mask = ants.morphology(__mask, operation='dilate', radius=__radius)
            __mask = ants.morphology(__mask, operation='erode', radius=__radius)
        
        
        if min_connected_component_size is not None:
            
            __size = min_connected_component_size / (__mask.spacing[0] * __mask.spacing[1])
            
            for rprop in skimage.measure.regionprops(skimage.measure.label(__mask.view())):
                if rprop['area'] < __size:
                    __mask[rprop['slice']] = 0
                    
                    
    return __mask.astype('uint8')
