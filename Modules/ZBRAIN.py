# ******************************************************************************
# This file contains functions to interact with the Z-BRAIN 1.0 dataset.
#
# This code shall be licensed under the GNU General Public License v3.0.
#
# (c) 2020- , Jan Mölter
# ******************************************************************************

__all__ = ['create_atlas',
           'list_anatomy_labels',
           'list_masks',
          ]


import os

import ants
#import nrrd
import numpy

import scipy.sparse

import json

import h5py


def create_atlas(files, anatomy_labels, masks, directory, spacing=None):
    """
    Creates a brain atlas with a custom set of anatomical labels and masks derived
    from the Z-BRAIN dataset.

    Parameters
    ----------
    files : dict
        Dictionary with fields `reference`, `anatomy labels database`,
        `mask database`, and `extensions` containing paths to the corresponding
        Z-BRAIN files.
    anatomy_labels : list of str
        Custom set of anatomical labels to be available in the brain atlas.
    masks : list of str
        Custom set of masks to be available in the brain atlas.
    directory : str
        Output path of the new brain atlas.
    spacing: (3,) tuple of float or int
        Spacing of the atlas (Units: 1µm). Default is None.

    Returns
    -------
    None
    """
    
    if not (type(files) is dict and 'reference' in files and 'anatomy labels database' in files and 'masks database' in files):
        raise TypeError('`files` is expected to be of type dict and to contain the keys `reference`, `anatomy labels database`, `mask database`, and `extensions` containing paths to the respective Z-BRAIN files.')
    
    for file in files.values():
        if type(file) is str:
            if not os.path.isfile(file):
                raise FileNotFoundError('The file {} does not exist.'.format(file))
        
            
    if type(anatomy_labels) is not dict:
        raise TypeError('`anatomy_labels` is expected to be of type dict.')
    if type(masks) is not dict:
        raise TypeError('`masks` is expected to be of type dict.')
    
    if os.path.isdir(directory):
        raise ValueError('The output directory (`directory`) already exists.')
    else:
        os.makedirs(directory)
        
        _atlas_directories = dict()
        for subdirectory in ['reference', 'labels', 'masks']:
            _atlas_directories[subdirectory] = os.path.join(directory, subdirectory)
            os.makedirs(_atlas_directories[subdirectory])
        
        
        __files_reference = extract_reference(files['reference'], os.path.join(_atlas_directories['reference'], 'reference.nrrd'), spacing=spacing)
        
        __reference_header_info = ants.image_header_info(files['reference'])
        
        __files_anatomy_labels = extract_anatomy_labels(files['anatomy labels database'], anatomy_labels, __reference_header_info, os.path.join(_atlas_directories['labels'], '{}.nrrd'), spacing=spacing)
        __files_masks = extract_masks(files['masks database'], masks, __reference_header_info, os.path.join(_atlas_directories['masks'], '{}.nrrd'), spacing=spacing)
        
        __files_anatomy_labels_extensions, __files_masks_extensions = extract_extensions(files['extensions'], {'labels': os.path.join(_atlas_directories['labels'], '{}.nrrd'), 'masks': os.path.join(_atlas_directories['masks'], '{}.nrrd')}, spacing=spacing)

        __files_anatomy_labels.update(__files_anatomy_labels_extensions)
        __files_masks.update(__files_masks_extensions)


        for files_dict in [__files_reference, __files_anatomy_labels, __files_masks]:
            for _ in files_dict.keys():
                files_dict[_] = os.path.relpath(files_dict[_], directory)
        
        content_dict = dict()
        content_dict['reference'] = __files_reference['reference']
        
        #content_dict['labels'] = dict()
        #for label, dataset in anatomy_labels.items():
        #    content_dict['labels'][label] = 'labels/{}.nrrd'.format(dataset)
        content_dict['labels'] = __files_anatomy_labels
        
        #content_dict['masks'] = dict()
        #for label, dataset in masks.items():
        #    content_dict['masks'][label] = 'masks/{}.nrrd'.format(dataset)
        content_dict['masks'] = __files_masks
        
        with open(os.path.join(directory, 'content.json'), 'w', encoding='utf-8') as _:
            json.dump(content_dict, _, ensure_ascii=False, indent=4)



def extract_reference(reference, output_file_format, spacing=None):
    """
    Extracts the anatomical reference image from the Z-BRAIN dataset.

    Parameters
    ----------
    reference : str
        Path to the reference image.
    output_file_format : str
        Output file.
    spacing: (3,) tuple of float or int
        Spacing of the atlas (Units: 1µm). Default is None.

    Returns
    -------
    None
    """

    if not os.path.isfile(reference):
        raise FileNotFoundError('The file {} does not exist.'.format(reference))
    
    __volume_image = ants.image_read(reference)
    
    if spacing is not None and __volume_image.spacing != spacing:
        __volume_image = ants.resample_image(__volume_image, spacing, use_voxels=False, interp_type=0)
        
    
    if not os.path.splitext(output_file_format)[1].upper() in ['.NRRD']:
        output_file += '.nrrd'
    
    __volume_image.to_file(output_file_format)
    
    
    del __volume_image
    
    return {'reference': output_file_format}


def list_anatomy_labels(database):
    """
    Lists the anatomical labels contained in the Z-BRAIN dataset.

    Parameters
    ----------
    database : str
        Path to the database of anatomical labels.

    Returns
    -------
    anatomy_labels : list of str
        Anatomical labels contained in the Z-BRAIN dataset.
    """

    if not os.path.isfile(database):
        raise FileNotFoundError('The file {} does not exist.'.format(database))
    
    with h5py.File(database, 'r') as hdf5_file:
        return [dataset for dataset in hdf5_file.keys()]

def extract_anatomy_labels(database, label_dict, reference_header_info, output_file_format, spacing=None):
    """
    ********************************************************************************
    Extracts a custom set of anatomical labels from the Z-BRAIN dataset.

    Parameters
    ----------
    database : str
        Path to the database of anatomical labels.
    label_dict : dict
        Dictionary of anatomical labels to extract.
    reference_header_info: dict
        Dictionary of corresponding image header information containing fields for
        `spacing`, `origin`, and `direction`.
    output_file_format : str
        Output file format, with placeholder `{}`.
    spacing: (3,) tuple of float or int
        Spacing of the atlas (Units: 1µm). Default is None.

    Returns
    -------
    None
    """

    if not os.path.isfile(database):
        raise FileNotFoundError('The file {} does not exist.'.format(database))

    __extracted_files = {}

    with h5py.File(database, 'r') as hdf5_file:
        for label, dataset in label_dict.items():
            if dataset not in hdf5_file.keys():
                raise ValueError('The anatomical label `{}` does not exist in the Z-BRAIN dataset.'.format(dataset))
            
            __volume_image_data = hdf5_file.get(dataset)[:]
            __volume_image_data = numpy.moveaxis(__volume_image_data, 0, -1)
            
            __volume_image_data = __volume_image_data.astype('float32')
        
        
            #__nrrd_header = dict()
            #__nrrd_header['encoding'] = 'raw'
            #__nrrd_header['space dimension'] = 3
            #__nrrd_header['space directions'] = numpy.diag(list(image_spacing))
            #__nrrd_header['space units'] = ['microns'] * 3
            #
            #nrrd.write('Data/Z-Brain/{}.nrrd'.format(dataset), __volume_image, index_order='C', header=__nrrd_header)
            
            
            __volume_image = ants.from_numpy(__volume_image_data, spacing=reference_header_info['spacing'], origin=reference_header_info['origin'], direction=reference_header_info['direction'])
            if spacing is not None and reference_header_info['spacing'] != spacing:
                __volume_image = ants.resample_image(__volume_image, spacing, use_voxels=False, interp_type=0)
            
            
            if not os.path.splitext(output_file_format)[1].upper() in ['.NRRD']:
                output_file_format += '.nrrd'
            
            __volume_image.to_file(output_file_format.format(dataset))

            __extracted_files[label] = output_file_format.format(dataset)
            
    
    del hdf5_file, label, dataset, __volume_image_data, __volume_image

    return __extracted_files


def list_masks(database):
    """
    Lists the masks contained in the Z-BRAIN dataset.

    Parameters
    ----------
    database : str
        Path to the database of masks.

    Returns
    -------
    masks : list of str
        Masks contained in the Z-BRAIN dataset.
    """

    if not os.path.isfile(database):
        raise FileNotFoundError('The file {} does not exist.'.format(database))

    with h5py.File(database, 'r') as hdf5_file:
        __MaskDatabaseNames = [hdf5_file[__ref__[0]][:].tobytes()[::2].decode() for __ref__ in hdf5_file['MaskDatabaseNames']]
        return __MaskDatabaseNames
        
def extract_masks(database, mask_dict, reference_header_info, output_file_format, spacing=None):
    """
    ********************************************************************************
    Extracts a custom set of masks from the Z-BRAIN dataset.

    Parameters
    ----------
    database : str
        Path to the database of masks.
    label_dict : dict
        Dictionary of masks to extract.
    reference_header_info: dict
        Dictionary of corresponding image header information containing fields for
        `spacing`, `origin`, and `direction`.
    output_file_format : str
        Output file format, with placeholder `{}`.
    spacing: (3,) tuple of float or int
        Spacing of the atlas (Units: 1µm). Default is None.

    Returns
    -------
    None
    """

    if not os.path.isfile(database):
        raise FileNotFoundError('The file {} does not exist.'.format(database))

    __extracted_files = {}

    with h5py.File(database, 'r') as hdf5_file:
        __DateCreated = hdf5_file['DateCreated'][:].tobytes()[::2].decode()
        
        __height = int(hdf5_file['height'][0,0])
        __width = int(hdf5_file['width'][0,0])
        __Zs = int(hdf5_file['Zs'][0,0])
        __MaskDatabaseNames = [hdf5_file[__ref__[0]][:].tobytes()[::2].decode() for __ref__ in hdf5_file['MaskDatabaseNames']]
        
        __shape = (__height * __width * __Zs, len(__MaskDatabaseNames))
        
        __MaskDatabase_data = hdf5_file['MaskDatabase/data'][:].astype('uint8')
        __MaskDatabase_ir = hdf5_file['MaskDatabase/ir'][:]
        __MaskDatabase_jc = hdf5_file['MaskDatabase/jc'][:]
        
        __MaskDatabase = scipy.sparse.csc_matrix((__MaskDatabase_data, __MaskDatabase_ir, __MaskDatabase_jc), shape=__shape)
        
        __MaskDatabaseOutlines_data = hdf5_file['MaskDatabaseOutlines/data'][:].astype('uint8')
        __MaskDatabaseOutlines_ir = hdf5_file['MaskDatabaseOutlines/ir'][:]
        __MaskDatabaseOutlines_jc = hdf5_file['MaskDatabaseOutlines/jc'][:]
        
        __MaskDatabaseOutlines = scipy.sparse.csc_matrix((__MaskDatabaseOutlines_data, __MaskDatabaseOutlines_ir, __MaskDatabaseOutlines_jc), shape=__shape)
        
        
    
    for label, dataset in mask_dict.items():
        if dataset not in __MaskDatabaseNames:
            raise ValueError('The mask `{}` does not exist in the Z-BRAIN dataset.'.format(dataset))
            
        __volume_image_data = numpy.asarray(__MaskDatabase[:,__MaskDatabaseNames.index(dataset)].todense()).reshape((__Zs, __width, __height))
        __volume_image_data = numpy.moveaxis(__volume_image_data, 0, -1).astype('uint8')
        
        #__volume_image_data[__volume_image_data == 0] = numpy.nan
        
        __volume_image = ants.from_numpy(__volume_image_data, spacing=reference_header_info['spacing'], origin=reference_header_info['origin'], direction=reference_header_info['direction'])
        if spacing is not None and reference_header_info['spacing'] != spacing:
            __volume_image = ants.resample_image(__volume_image, spacing, use_voxels=False, interp_type=0)
        
        
        if not os.path.splitext(output_file_format)[1].upper() in ['.NRRD']:
            output_file_format += '.nrrd'
        
        __volume_image.to_file(output_file_format.format(dataset))

        __extracted_files[label] = output_file_format.format(dataset)
    
    
    
    #del hdf5_file, label, dataset, __volume_image_data, __volume_image
    #del __DateCreated, __height, __width, __Zs, __MaskDatabaseNames, __shape, __MaskDatabase_data, __MaskDatabase_ir, __MaskDatabase_jc, __MaskDatabase, __MaskDatabaseOutlines_data, __MaskDatabaseOutlines_ir, __MaskDatabaseOutlines_jc, __MaskDatabaseOutlines

    return __extracted_files

def extract_extensions(extensions, output_file_format, spacing=None):
    """
    Extracts extensions from the Z-BRAIN dataset.

    Parameters
    ----------

    Returns
    -------
    None
    """

    __extracted_files = {}
    __extracted_files['labels'] = {}
    __extracted_files['masks'] = {}

    for x_type in ['labels', 'masks']:
        if x_type in extensions:
            for label, dataset in extensions[x_type].items():

                if not os.path.isfile(dataset):
                    raise FileNotFoundError('The file {} does not exist.'.format(dataset))
    
                __volume_image = ants.image_read(dataset)
                if x_type == 'masks':
                    __volume_image = __volume_image.astype('uint8')
    
                if spacing is not None and __volume_image.spacing != spacing:
                    __volume_image = ants.resample_image(__volume_image, spacing, use_voxels=False, interp_type=0)
        
    
                if not os.path.splitext(output_file_format[x_type])[1].upper() in ['NRRD']:
                    output_file_format[x_type] += '.nrrd'


                __fname = os.path.splitext(os.path.basename(dataset))[0]
    
                __volume_image.to_file(output_file_format[x_type].format(__fname))
    
                __extracted_files[x_type][label] = output_file_format[x_type].format(__fname)
    
                del __volume_image
    
    return __extracted_files['labels'], __extracted_files['masks']
