import os
import sys

import math

import numpy

import io
import json

import base64

import tifffile
import PIL

from Modules import VolumeImagery


def segmentation_export(moving_data_directory, moving_image, output_file_format, export_format, masks=None, axis=2, right_hemisphere_mask=None, verbose=None):

    if not os.path.isdir(moving_data_directory):
        raise FileNotFoundError('Moving image data directory does not exist.')
    if not os.path.isfile(os.path.join(moving_data_directory, moving_image)):
        raise FileNotFoundError('Moving image does not exist.')
    if not os.path.isdir(os.path.join(moving_data_directory, 'region-masks')):
        raise FileNotFoundError('Region masks directory `region-masks` does not exist in moving image data directory.')
    if not os.path.isfile(os.path.join(moving_data_directory, 'region-masks', 'content.json')):
        raise FileNotFoundError('Region masks content file does not exist.')
        
    if not export_format.lower() in ['labelme']:
        raise ValueError('Export format ''{}'' is not supported.'.format(export_format))
    
    
    
    if verbose is not None and verbose:
        print('=== {} ==='.format('Load moving image region masks'), file=sys.stdout)
        print(' Directory: {}'.format(os.path.join(moving_data_directory, 'region-masks')), file=sys.stdout)
        print('', file=sys.stdout)
    
    with open(os.path.join(moving_data_directory, 'region-masks', 'content.json'), 'r', encoding='utf-8') as _:
        moving_image_masks_dict = json.load(_)
    
    
    
    find_mask_contours__kwargs = {
        'axis': axis,
        'segment_length': 10,
        'use_voxels': False,
        'include_holes': True,
    }
    
    if export_format.lower() == 'labelme':
        find_mask_contours__kwargs['include_holes'] = False
    
    
    
    
    __RH_mask_image = None
    if not right_hemisphere_mask is None and right_hemisphere_mask in moving_image_masks_dict:
        #if verbose is not None and verbose:
        #    print(' - {}'.format(right_hemisphere_mask), file=sys.stdout)
            
        __RH_mask_image = VolumeImagery.load(os.path.join(moving_data_directory, 'region-masks', moving_image_masks_dict.pop(right_hemisphere_mask)), pixeltype='unsigned char')
    
    if masks is None:
        masks = list(moving_image_masks_dict.keys())
    
    region_contours = dict()
    
    
    for mask in masks:
        if not mask in moving_image_masks_dict:
            continue
            
        if verbose is not None and verbose:
            print(' - {}'.format(mask), file=sys.stdout)
            
        __mask_image = VolumeImagery.load(os.path.join(moving_data_directory, 'region-masks', moving_image_masks_dict[mask]), pixeltype='unsigned char')
        
        if __RH_mask_image is None:
            region_contours[mask] = VolumeImagery.find_mask_contours(__mask_image, **find_mask_contours__kwargs)
        else:
            __mask_image_RH = __mask_image.new_image_like(numpy.bitwise_and(__RH_mask_image.view(), __mask_image.view()))
            region_contours['{} (right)'.format(mask)] = VolumeImagery.find_mask_contours(__mask_image_RH, **find_mask_contours__kwargs)
            del __mask_image_RH
            
            __mask_image_LH = __mask_image.new_image_like(numpy.bitwise_and(1 - __RH_mask_image.view(), __mask_image.view()))
            region_contours['{} (left)'.format(mask)] = VolumeImagery.find_mask_contours(__mask_image_LH, **find_mask_contours__kwargs)
            del __mask_image_LH
            
        del __mask_image
    
    if not __RH_mask_image is None:
        del __RH_mask_image
        
    if verbose is not None and verbose:
        print('', file=sys.stdout)
    
    if verbose is not None and verbose:
        print('=== {} ==='.format('Load moving image'), file=sys.stdout)
        print(' Directory: {}'.format(moving_data_directory), file=sys.stdout)
        print('', file=sys.stdout)
    
    reference_image = VolumeImagery.load(os.path.join(moving_data_directory, moving_image))
    reference_image = (reference_image - reference_image.min()) / (reference_image.max() - reference_image.min())
    
    
    if verbose is not None and verbose:
        print('=== {} ==='.format('Export segmentation images'), file=sys.stdout)
        
    if export_format.lower() == 'labelme':
        labelme_segmentation_export(reference_image, region_contours, axis=axis, output_file_format=output_file_format, verbose=verbose)

        
def labelme_segmentation_export(reference_image, region_contours, axis, output_file_format, verbose=None):
    
    def array_to_b64image(image_array, format='PNG'):
        
        __image = PIL.Image.fromarray(numpy.uint8(image_array * 255))
        __buffer = io.BytesIO()
        __image.save(__buffer, format=format)
        __buffer.seek(0)
        
        return base64.b64encode(__buffer.getvalue()).decode()
    
    
    reference_image_stack = VolumeImagery.to_image_stack(reference_image, axis=axis)
    
    
    __counter_length = math.ceil(math.log10(len(reference_image_stack)))
    __files = []
    
    for i in range(len(reference_image_stack)):
        
        __shapes = []
        for mask in list(region_contours.keys()):
            
            for C in region_contours[mask][i]:
                __shapes += [(mask, [list(_) for _ in list(C)])]
        

        labelme_data = {
            'version': '4.5.6',
            'flags': {},
            'shapes': [
                {
                    'label': shape[0],
                    'points': shape[1],
                    'group_id': None,
                    'shape_type': 'polygon',
                    'flags': {}
                }
                for shape in __shapes],
            
            'imagePath': '',
            'imageData': array_to_b64image(reference_image_stack[i].T, format='PNG'),
            'imageHeight': reference_image_stack[i].shape[0],
            'imageWidth': reference_image_stack[i].shape[1],
        }
        
        __files += [output_file_format.format(('{:0'+ str(__counter_length) +'}').format(i))]
        
        if verbose is not None and verbose:
            print(' - Plane {}'.format(i), file=sys.stdout)
        
        with open(__files[-1], 'w') as _:
            json.dump(labelme_data, _)
    
    if verbose is not None and verbose:
        print('', file=sys.stdout)
    
    return __files


if __name__ == "__main__":
    # ********************************************************************************
    # Usage:
    # > python segmentationExport.py --moving-data-directory <...>
    #                           	 --moving-image <...>
    #                                --output-file-format <...>
    #                                --export-format <...>
    #                                --masks <...>
    #                                --axis <...>
    #	                             --right-hemisphere-mask <...>
    #
    
    import argparse
    
    
    __parser = argparse.ArgumentParser()
    __parser.add_argument('--moving-data-directory', dest='moving_data_directory', type=str, required=True)
    __parser.add_argument('--moving-image', dest='moving_image', type=str, required=True)
    __parser.add_argument('--output-file-format', dest='output_file_format', type=str, required=True)
    __parser.add_argument('--export-format', dest='export_format', type=str, required=True)
    __parser.add_argument('--masks', dest='masks', nargs='+', default=None, type=str, required=False)
    __parser.add_argument('--axis', dest='axis', default=2, type=int, required=False)
    __parser.add_argument('--right-hemisphere-mask', default=None, type=str, required=False)
    
    kwargs = vars(__parser.parse_args())

    # ********************************************************************************
    # Include default parameters

    # ********************************************************************************
    # Execute main function

    try:
        segmentation_export(**kwargs, verbose=True)
    except:
        print('An error occured. Operation could not be completed.', file=sys.stderr)
        sys.exit(1)

    sys.exit(0)
