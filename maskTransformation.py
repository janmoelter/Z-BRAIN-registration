import os
import sys

import traceback

import json

import ants

from Modules import BrainAtlas
from Modules import VolumeImagery


def mask_transformation(reference_atlas_directory, atlas_masks, moving_data_directory, moving_image, optimisation_args=None, verbose=None):

    if not os.path.isdir(reference_atlas_directory):
        raise FileNotFoundError('Reference atlas directory does not exist.')
    if not os.path.isdir(moving_data_directory):
        raise FileNotFoundError('Moving image data directory does not exist.')
    if not os.path.isfile(os.path.join(moving_data_directory, moving_image)):
        raise FileNotFoundError('Moving image does not exist.')
    
    
    # Step 1: Load data

    if verbose is not None and verbose:
        print('=== {} ==='.format('Load reference atlas'), file=sys.stdout)
        print(' Directory: {}'.format(reference_atlas_directory), file=sys.stdout)
        print('', file=sys.stdout)
    
    __reference_atlas = BrainAtlas.load(reference_atlas_directory, labels=[], masks=atlas_masks)

    if verbose is not None and verbose:
        print('=== {} ==='.format('Load moving image'), file=sys.stdout)
        print(' Directory: {}'.format(moving_data_directory), file=sys.stdout)
        print('', file=sys.stdout)

    __moving_image = ants.image_read(os.path.join(moving_data_directory, moving_image))
    
    # Step 2: Warp atlas masks

    if verbose is not None and verbose:
        print('=== {} ==='.format('Transform masks of the reference atlas'), file=sys.stdout)
    
    if not os.path.isdir(os.path.join(moving_data_directory, 'region-masks')):
        os.mkdir(os.path.join(moving_data_directory, 'region-masks'))
    
    
    # ********************************************************************************

    content_dict = dict()
    
    for maskname in __reference_atlas.list_masks():
        
        __regularised_maskname = maskname.replace('::', '').replace('  ', '__').replace(' ', '-')
        
        kwargs = {
            'fixed': __moving_image,
            'moving': __reference_atlas.masks[maskname].astype('float32'),
            'transformlist': os.path.join(moving_data_directory, 'registration/transformation_InverseComposite.h5'),
            'interpolator': 'linear'
        }

        if verbose is not None and verbose:
            print(' - {}'.format(maskname), file=sys.stdout)
    
        __warped_mask = ants.apply_transforms(**kwargs).astype('uint8')
        
        if optimisation_args is not None:
            __warped_mask = VolumeImagery.mask_optimisation(__warped_mask, **optimisation_args)
        
        content_dict[maskname] = __regularised_maskname + '.nrrd'
        
        __warped_mask.to_file(os.path.join(moving_data_directory, 'region-masks', content_dict[maskname]))

    with open(os.path.join(moving_data_directory, 'region-masks', 'content.json'), 'w', encoding='utf-8') as _:
        json.dump(content_dict, _, ensure_ascii=False, indent=4)
    
    if verbose is not None and verbose:
        print('', file=sys.stdout)


if __name__ == "__main__":
    # ********************************************************************************
    # Argument parsing
    #
    
    import argparse
    
    
    __parser = argparse.ArgumentParser(
        description='Transforms a set of region masks from a reference atlas back onto a volume image using the transformation obtained from a previous registration of that image to the atlas.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    __parser.add_argument('--reference-atlas-directory', dest='reference_atlas_directory', type=str, required=True, metavar='<path>', help='Path to the directory of a reference atlas.')
    __parser.add_argument('--atlas-masks', dest='atlas_masks', default=None, nargs='*', type=str, required=False, metavar='<name>', help='Names of region masks defined in the reference atlas. These masks will be transformed back to obtain masks for the corresponding regions on the moving image. If this argument is not given, all masks of the reference atlas will be transformed.')
    __parser.add_argument('--moving-data-directory', dest='moving_data_directory', type=str, required=True, metavar='<path>', help='Path to the directory of a moving image.')
    __parser.add_argument('--moving-image', dest='moving_image', type=str, required=True, metavar='<path>', help='Moving image in the directory of the moving image. This image must have been previously registered to the reference atlas.')
    __parser.add_argument('--no-optimisation', action='store_true', required=False, help='Indicates whether to optimise the masks after the transformation. The optimisation includes a dilation-erosion procedure and removes small connected components. This generally improves the result but might require more memory.')
    
    kwargs = vars(__parser.parse_args())

    # ********************************************************************************
    # Preprocess arguments

    if not kwargs['no_optimisation']:
        kwargs['optimisation_args'] = {
            'dilation_erosion_radius' : 20,
            'min_connected_component_size' : 20**3
        }

    else:
        kwargs['optimisation_args'] = None
    
    kwargs.pop('no_optimisation')

    # ********************************************************************************
    # Execute main function

    try:
        mask_transformation(**kwargs, verbose=True)
    except:
        print('', file=sys.stdout)
        print('', file=sys.stdout)
        print('An error occured. Operation could not be completed.', file=sys.stdout)
        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1)

    sys.exit(0)
