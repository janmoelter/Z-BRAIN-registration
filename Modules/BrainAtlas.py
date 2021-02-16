# ******************************************************************************
# This file contains the definition of a custom object for a brain atlas and
# functions to interact with that.
#
# This code shall be licensed under the GNU General Public License v3.0.
#
# (c) 2020- , Jan Mölter
# ******************************************************************************

__all__ = ['BrainAtlas',
           'load',
           'export',
          ]


import os

import json

import ants
import numpy


class BrainAtlas(object):
    def __init__(self, reference, labels, masks):
        if not (reference is None or type(reference) is ants.core.ants_image.ANTsImage):
            raise TypeError('`reference` is expected to be either of type ants.core.ants_image.ANTsImage or None.')
        
        if not type(labels) is dict:
            raise ValueError('`labels` is expected to be a dictionary.')
        for ___key, ___value in labels.items():
            if not type(___value) is ants.core.ants_image.ANTsImage:
                raise TypeError('`labels`[\'{}\'] is expected to be of type ants.core.ants_image.ANTsImage.'.format(__key))
        
        if not type(masks) is dict:
            raise ValueError('`masks` is expected to be a dictionary.')
        for ___key, ___value in masks.items():
            if not type(___value) is ants.core.ants_image.ANTsImage:
                raise TypeError('`masks`[\'{}\'] is expected to be of type ants.core.ants_image.ANTsImage.'.format(__key))
        
        if reference is None and len(labels) == 0 and len(masks) == 0:
            raise ValueError('Either `reference`, `labels`, or `masks` is expected to contain at least 1 image.')
        
        self.__reference = reference
        self.__labels = labels
        self.__masks = masks
        
        self.__set_properties()
        
    def __del__(self):
        del self.__reference
        del self.__labels
        del self.__masks
    
    def __repr__(self):
        __repr__ = 'Brain Atlas\n' +\
                   ' - {:<10} : {}\n'.format('Dimension', self.dimensions)+\
                   ' - {:<10} : {}\n'.format('Spacing', self.spacing)
        
        if len(self.__labels) > 0:
            __repr__ = __repr__ +\
                       '\n'+\
                       ' - Anatomy labels:\n'+\
                       '\n'.join(['   * {}'.format(label) for label in self.list_labels()])
            
        if len(self.__masks) > 0:
            __repr__ = __repr__ +\
                       '\n'+\
                       ' - Masks:\n'+\
                       '\n'.join(['   * {}'.format(mask) for mask in self.list_masks()])
        
        
        return  __repr__
        
    @property
    def dimensions(self):
        """
        Returns the dimensions of the brain atlas.

        Parameters
        ----------
        None

        Returns
        -------
        dimensions : (?,) tuple
            Dimensions (shape) of the brain atlas.
        """
        return self.__dimensions
    
    @property
    def spacing(self):
        """
        Returns the spacing of the brain atlas.

        Parameters
        ----------
        None

        Returns
        -------
        spacing : (?,) tuple
            Spacing of the brain atlas (Units: 1µm).
        """
        return self.__spacing
    
    @property
    def reference(self):
        """
        Returns the reference image of the brain atlas.

        Parameters
        ----------
        None

        Returns
        -------
        reference : ants.core.ants_image.ANTsImage or None
            Reference image of the brain atlas.
        """
        return self.__reference
    
    @property
    def labels(self):
        """
        Returns the anatomical label images of the brain atlas.

        Parameters
        ----------
        None

        Returns
        -------
        labels : dict
            Anatomical label images of the brain atlas.
        """
        return self.__labels
    
    def list_labels(self):
        """
        Lists the anatomical label images ín the brain atlas.

        Parameters
        ----------
        None

        Returns
        -------
        labels : list
            Names of anatomical label images in the brain atlas.
        """
        return list(self.__labels.keys())
    
    @property
    def masks(self):
        """
        Returns the mask images of the brain atlas.

        Parameters
        ----------
        None

        Returns
        -------
        masks : dict
            Mask images of the brain atlas.
        """
        return self.__masks
    
    def list_masks(self):
        """
        Lists the mask images ín the brain atlas.

        Parameters
        ----------
        None

        Returns
        -------
        labels : list
            Names of mask images in the brain atlas.
        """
        return list(self.__masks.keys())

    def resample(self, spacing):
        """
        Resamples the entire brain atlas.

        Parameters
        ----------
        spacing : tuple of float
            Spacing to which the brain atlas is resampled.

        Returns
        -------
        None
        """

        if self.__reference is not None:
            self.__reference = ants.resample_image(self.__reference, spacing=spacing, use_voxels=False, interp_type=0)
        
        for k in self.__labels.keys():
            self.__labels[k] = ants.resample_image(self.__labels[k], spacing=spacing, use_voxels=False, interp_type=0)
            
        for k in self.__masks.keys():
            self.__masks[k] = ants.resample_image(self.__masks[k], spacing=spacing, use_voxels=False, interp_type=0)
    
    def crop(self, lower_mark, upper_mark):
        """
        Crops the brain atlas.

        Parameters
        ----------
        lower_mark : tuple of float
        upper_mark : tuple of float

        Returns
        -------
        None
        """

        lower_index = tuple(numpy.floor(numpy.array(lower_mark) / self.__spacing).astype('int'))
        upper_index = tuple(numpy.array(self.__dimensions) - numpy.floor(numpy.array(upper_mark) / self.__spacing).astype('int'))
        
        self.__crop_indices(lower_index, upper_index)
        self.__set_properties()
        
    def add_masks(self, masks):
        """
        Adds a set of masks contained in the atlas.

        Parameters
        ----------
        masks : list of str
            Masks to be added.

        Returns
        -------
        mask_sum : ants.core.ants_image.ANTsImage
            Reference image of the brain atlas.
        """

        __mask_data = numpy.zeros(self.__dimensions, dtype='uint8')

        for mask in masks:
            __mask_data = numpy.logical_or(__mask_data, self.__masks[mask][:,:,:]).astype('uint8')
        
        return self.__masks[masks[0]].new_image_like(__mask_data)
    
    def __crop_indices(self, lower_index, upper_index):
        if self.__reference is not None:
            self.__reference = ants.crop_indices(self.__reference, lower_index, upper_index)
        
        for k in self.__labels.keys():
            self.__labels[k] = ants.crop_indices(self.__labels[k], lower_index, upper_index)
            
        for k in self.__masks.keys():
            self.__masks[k] = ants.crop_indices(self.__masks[k], lower_index, upper_index)

    def __set_properties(self):
        __image = None
        
        if self.__reference is not None:
            __image = self.__reference
        else:
            if len(self.__labels) > 0:
                __image = next(iter(self.__labels.values()))
            else:
                if len(self.__masks) > 0:
                    __image = next(iter(self.__masks.values()))
                else:
                    None
        
        self.__dimensions = __image.shape
        self.__spacing = __image.spacing
            
    
def load(path, labels=None, masks=None):
    """
    Loads a brain atlas from a directory.

    Parameters
    ----------
    path : str
        Path of the brain atlas. This directory is expected to have a content files
        with reference to all available files.
    labels : list of str
        Anatomical labels to be loaded into the brain atlas. If not given, every
        anatomical label image is loaded. Default is None.
    masks : list of str
        Masks to be loaded into the brain atlas. If not given, every mask image is
        loaded. Default is None.

    Returns
    -------
    reference : BrainAtlas.BrainAtlas
        Brain atlas.
    """
    
    __atlas_path = path
    if not os.path.isdir(__atlas_path):
        raise FileNotFoundError('The directory {} does not exist.'.format(__atlas_path))
    if not os.path.isfile(os.path.join(__atlas_path, 'content.json')):
        raise FileNotFoundError('The directory does not contain the file content.json.')
    
    with open(os.path.join(__atlas_path, 'content.json'), 'r', encoding='utf-8') as json_file:
        content_dict = json.load(json_file)
        
    if content_dict['reference'] is not None:
        if not os.path.isfile(os.path.join(__atlas_path, content_dict['reference'])):
            raise FileNotFoundError('The reference image at {} does not exist.'.format(content_dict['reference']))
        __reference = ants.image_read(os.path.join(__atlas_path, content_dict['reference']), pixeltype='float')
    else:
        __reference = None

    __labels = dict()
    for label in content_dict['labels'].keys():
        if labels is None or (labels is not None and label in labels):
            if os.path.isfile(os.path.join(__atlas_path, content_dict['labels'][label])):
                __labels[label] = ants.image_read(os.path.join(__atlas_path, content_dict['labels'][label]), pixeltype='float')
            else:
                raise FileNotFoundError('The label image at {} does not exist.'.format(content_dict['labels'][label]))
            
    __masks = dict()
    for mask in content_dict['masks'].keys():
        if masks is None or (masks is not None and mask in masks):
            if os.path.isfile(os.path.join(__atlas_path, content_dict['masks'][mask])):
                __masks[mask] = ants.image_read(os.path.join(__atlas_path, content_dict['masks'][mask]), pixeltype='unsigned char')
            else:
                raise FileNotFoundError('The mask image at {} does not exist.'.format(content_dict['masks'][mask]))
    
    
    return BrainAtlas(__reference, __labels, __masks)

def export(brain_atlas, labels=None, masks=None):
    raise NotImplementedError('This function has not be implemented yet.')
