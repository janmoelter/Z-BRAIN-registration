import os
import sys

import traceback

import math

import numpy

import io
import json

import matplotlib
import matplotlib.pyplot as plt

import base64

import tifffile
import PIL

from Modules import VolumeImagery


def segmentation_export(moving_data_directory, moving_image, image_order, image_orientation, output_file_format, export_format, masks=None, right_hemisphere_mask=None, verbose=None):

    if not os.path.isdir(moving_data_directory):
        raise FileNotFoundError('Moving image data directory does not exist.')
    if not os.path.isfile(os.path.join(moving_data_directory, moving_image)):
        raise FileNotFoundError('Moving image does not exist.')
    if not os.path.isdir(os.path.join(moving_data_directory, 'region-masks')):
        raise FileNotFoundError('Region masks directory `region-masks` does not exist in moving image data directory.')
    if not os.path.isfile(os.path.join(moving_data_directory, 'region-masks', 'content.json')):
        raise FileNotFoundError('Region masks content file does not exist.')
    
    export_format = export_format.lower()
    if not export_format in ['image', 'labelme']:
        raise ValueError('Export format ''{}'' is not supported.'.format(export_format))
        
    #if not ((type(export_format) is list and all([type(_) is str for _ in export_format])) or type(export_format) is str):
    #    raise TypeError('`export_format` is expected to either be of type list and to contain elements of type str or to be of type str.')
    #
    #export_format = [_.lower() for _ in list(export_format)]
    #    
    #if not all([_ in ['image', 'labelme'] for _ in export_format]):
    #    raise ValueError('Export format ''{}'' is not supported.'.format(export_format))
    
    if verbose is not None and verbose:
        print('=== {} ==='.format('Load moving image'), file=sys.stdout)
        print(' Directory: {}'.format(moving_data_directory), file=sys.stdout)
        print('', file=sys.stdout)
    
    reference_image = VolumeImagery.load(os.path.join(moving_data_directory, moving_image))
    reference_image = VolumeImagery.normalise(reference_image)
    
    
    
    if verbose is not None and verbose:
        print('=== {} ==='.format('Load moving image region masks'), file=sys.stdout)
        print(' Directory: {}'.format(os.path.join(moving_data_directory, 'region-masks')), file=sys.stdout)
        print('', file=sys.stdout)
    
    with open(os.path.join(moving_data_directory, 'region-masks', 'content.json'), 'r', encoding='utf-8') as _:
        moving_image_masks_dict = json.load(_)
    
    
    
    find_mask_contours__kwargs = {
        'segment_length': 0,
        'use_voxels': False,
        'include_holes': True,
    }
    
    if export_format == 'image':
        find_mask_contours__kwargs['segment_length'] = 5
    
    if export_format == 'labelme':
        find_mask_contours__kwargs['segment_length'] = 10
        find_mask_contours__kwargs['include_holes'] = False
    
    
    
    
    __RH_mask_image = None
    if not right_hemisphere_mask is None and right_hemisphere_mask in moving_image_masks_dict:
        #if verbose is not None and verbose:
        #    print(' - {}'.format(right_hemisphere_mask), file=sys.stdout)
            
        __RH_mask_image = VolumeImagery.load(os.path.join(moving_data_directory, 'region-masks', moving_image_masks_dict.pop(right_hemisphere_mask)), pixeltype='unsigned char')
        __RH_mask_image = __RH_mask_image.reorient_image2(orientation=reference_image.orientation)
    
    if masks is None:
        masks = list(moving_image_masks_dict.keys())
    
    region_contours = dict()
    
    
    for mask in masks:
        if not mask in moving_image_masks_dict:
            continue
            
        if verbose is not None and verbose:
            print(' - {}'.format(mask), file=sys.stdout)
            
        __mask_image = VolumeImagery.load(os.path.join(moving_data_directory, 'region-masks', moving_image_masks_dict[mask]), pixeltype='unsigned char')
        __mask_image = __mask_image.reorient_image2(orientation=reference_image.orientation)
        
        if __RH_mask_image is None:
            __image_stack, __image_stack_spacing = VolumeImagery.to_image_stack(__mask_image, stack_orientation=image_order, image_orientation=image_orientation, return_spacing=True)

            region_contours[mask] = VolumeImagery.find_mask_contours(__image_stack, __image_stack_spacing, **find_mask_contours__kwargs)
        else:
            region_contours[mask] = {}
            
            __mask_image_RH = __mask_image.new_image_like(numpy.bitwise_and(__RH_mask_image.view(), __mask_image.view()))
            __image_stack, __image_stack_spacing = VolumeImagery.to_image_stack(__mask_image_RH, stack_orientation=image_order, image_orientation=image_orientation, return_spacing=True)
            del __mask_image_RH
            region_contours[mask]['right'] = VolumeImagery.find_mask_contours(__image_stack, __image_stack_spacing, **find_mask_contours__kwargs)
            
            __mask_image_LH = __mask_image.new_image_like(numpy.bitwise_and(1 - __RH_mask_image.view(), __mask_image.view()))
            __image_stack, __image_stack_spacing = VolumeImagery.to_image_stack(__mask_image_LH, stack_orientation=image_order, image_orientation=image_orientation, return_spacing=True)
            del __mask_image_LH
            region_contours[mask]['left'] = VolumeImagery.find_mask_contours(__image_stack, __image_stack_spacing, **find_mask_contours__kwargs)
            
            
        del __mask_image
        del __image_stack
        del __image_stack_spacing
    
    if not __RH_mask_image is None:
        del __RH_mask_image
        
    if verbose is not None and verbose:
        print('', file=sys.stdout)
    
    if verbose is not None and verbose:
        print('=== {} ==='.format('Export segmentation images'), file=sys.stdout)

    reference_image_stack, reference_image_spacing = VolumeImagery.to_image_stack(reference_image, stack_orientation=image_order, image_orientation=image_orientation, return_spacing=True)
    
    if export_format.lower() == 'image':
        if not os.path.splitext(output_file_format)[1].upper() in ['.PNG']:
            output_file_format += '.png'

        reference_image_stack_aspect_ratio = (reference_image_spacing[2] * reference_image_stack[0].shape[1]) / (reference_image_spacing[1] * reference_image_stack[0].shape[0])
        
        return png_segmentation_export(reference_image_stack, region_contours, output_file_format=output_file_format, aspect_ratio=reference_image_stack_aspect_ratio, contour_cmap='plasma', verbose=verbose)
        
    if export_format.lower() == 'labelme':
        if not os.path.splitext(output_file_format)[1].upper() in ['.JSON']:
            output_file_format += '.json'
            
        return labelme_segmentation_export(reference_image_stack, region_contours, output_file_format=output_file_format, verbose=verbose)


def labelme_segmentation_export(reference_image_stack, region_contours, output_file_format, verbose=None):
    
    def array_to_b64image(image_array, format='PNG'):
        
        __image = PIL.Image.fromarray(numpy.uint8(image_array * 255))
        __buffer = io.BytesIO()
        __image.save(__buffer, format=format)
        __buffer.seek(0)
        
        return base64.b64encode(__buffer.getvalue()).decode()
    
    
    __counter_length = math.ceil(math.log10(len(reference_image_stack)))
    __files = []
    
    for i in range(len(reference_image_stack)):
        
        __shapes = []
        for mask in list(region_contours.keys()):
            
            if type(region_contours[mask]) is dict:
                for mask_subkey in list(region_contours[mask].keys()):
                    for C in region_contours[mask][mask_subkey][i]:
                        __shapes += [('{} ({})'.format(mask, mask_subkey), [list(_) for _ in list(C[:,[1,0]])])]
            else:
                for C in region_contours[mask][i]:
                    __shapes += [('{}'.format(mask), [list(_) for _ in list(C[:,[1,0]])])]
        

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
            'imageData': array_to_b64image(reference_image_stack[i], format='PNG'),
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

def png_segmentation_export(reference_image_stack, region_contours, output_file_format, aspect_ratio=1, contour_cmap='plasma', verbose=None):
    
    __counter_length = math.ceil(math.log10(len(reference_image_stack)))
    __files = []
    
    __figure_width_cm = 10
    
    __figure_contour_colors = matplotlib.cm.get_cmap(contour_cmap)(numpy.linspace(0, 1, len(region_contours.keys())))
    
    for i in range(len(reference_image_stack)):
        
        __figure = matplotlib.pyplot.figure(frameon=False);
        __figure.set_size_inches(__figure_width_cm * 1/2.54, (__figure_width_cm/aspect_ratio) * 1/2.54);
        
        __axis = matplotlib.pyplot.Axes(__figure, [0., 0., 1., 1.]);
        __axis.set_axis_off();
        
        __figure.add_axes(__axis);
        
        __axis.imshow(reference_image_stack[i], aspect='auto', cmap='gray', interpolation='bicubic');

        
        
        
        for m, mask in enumerate(region_contours.keys()):
            if type(region_contours[mask]) is dict:
                for mask_subkey in list(region_contours[mask].keys()):
                    for C in region_contours[mask][mask_subkey][i]:
                        __axis.plot(C[:,1], C[:,0], color=__figure_contour_colors[m]);
            else:
                for C in region_contours[mask][i]:
                    __axis.plot(C[:,1], C[:,0], color=__figure_contour_colors[m]);
        
        
        __files += [output_file_format.format(('{:0'+ str(__counter_length) +'}').format(i))]

        if verbose is not None and verbose:
            print(' - Plane {}'.format(i), file=sys.stdout)
        
        __figure.savefig(__files[-1], dpi=300);
        
        matplotlib.pyplot.close(__figure);
    
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
    __parser.add_argument('--plane-order', dest='image_order', default='I', type=str, required=False)
    __parser.add_argument('--plane-orientation', dest='image_orientation', nargs='+', default=['A', 'L'], type=str, required=False)
    __parser.add_argument('--output-file-format', dest='output_file_format', type=str, required=True)
    __parser.add_argument('--export-format', dest='export_format', type=str, required=True)
    __parser.add_argument('--masks', dest='masks', nargs='+', default=None, type=str, required=False)
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
        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1)

    sys.exit(0)
