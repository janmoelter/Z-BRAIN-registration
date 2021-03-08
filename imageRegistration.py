import os
import sys

import traceback

import json

import numpy

import ants
from Modules import antsX

from Modules import BrainAtlas


def registration(reference_atlas_directory, atlas_registration_label, moving_data_directory, moving_image, initialisation=False, verbose=None):

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
    
    __reference_atlas = BrainAtlas.load(reference_atlas_directory, labels=[atlas_registration_label], masks=[])

    if verbose is not None and verbose:
        print('=== {} ==='.format('Load moving image'), file=sys.stdout)
        print(' Directory: {}'.format(moving_data_directory), file=sys.stdout)
        print('', file=sys.stdout)

    __moving_image = ants.image_read(os.path.join(moving_data_directory, moving_image))

    
    # Step 2: Perform registration

    if verbose is not None and verbose:
        print('=== {} ==='.format('Perform image registration'), file=sys.stdout)
    
    if not os.path.isdir(os.path.join(moving_data_directory, 'registration')):
        os.mkdir(os.path.join(moving_data_directory, 'registration'))

    # ********************************************************************************
    # Find initial transformation
    #

    #if initialisation:
    #
    #    if verbose is not None and verbose:
    #        print('- {}'.format('Find initial registration transformation'), file=sys.stdout)
    #    
    #    kwargs = {
    #        'image1': ants.resample_image(__reference_atlas.labels[atlas_registration_label], resample_params=(5,5,5)),
    #        'image2': ants.resample_image(__moving_image, resample_params=(5,5,5)),
    #        'transform': 'Affine',
    #        'local_search_iterations': 10,
    #        'thetas': numpy.linspace(-5, 5, 5),
    #        'thetas2': numpy.linspace(-5, 5, 5),
    #        'thetas3': numpy.linspace(-5, 5, 5),
    #        'txfn': os.path.join(moving_data_directory, 'registration', 'initialization.mat'),
    #    }
    #    
    #    ants.invariant_image_similarity(**kwargs)
    
    # ********************************************************************************
    # SyNQuick Registration
    #
    # This registration procedure is used by Favre-Bulle et al. 2018 (Curr. Biol. 28).
    #

    if verbose is not None and verbose:
        print('- {}'.format('Find complete registration transformation'), file=sys.stdout)
        
    kwargs = {
        'fixed': __reference_atlas.labels[atlas_registration_label],
        'moving': __moving_image,
        'type_of_transform': 'antsRegistrationSyNQuick[s]',
        'write_composite_transform': True,
        'outprefix': os.path.join(moving_data_directory, 'registration', 'transformation_'),
        'verbose': True
    }

    #if initialisation and os.path.isfile(os.path.join(moving_data_directory, 'registration', 'initialization.mat')):
    #    kwargs['initial_transform'] = os.path.join(moving_data_directory, 'registration', 'initialization.mat')
    
    ants.registration(**kwargs)

    if verbose is not None and verbose:
        print('', file=sys.stdout)
    
    if verbose is not None and verbose:
        print('=== {} ==='.format('Write registration transformation header'), file=sys.stdout)
    
    header_dict = {}

    header_dict['fixed'] = antsX.header_info(__reference_atlas.labels[atlas_registration_label])
    header_dict['fixed']['direction'] = header_dict['fixed']['direction'].tolist()
    
    header_dict['moving'] = antsX.header_info(__moving_image)
    header_dict['moving']['direction'] = header_dict['moving']['direction'].tolist()

    with open(os.path.join(moving_data_directory, 'registration', 'transformation_header.json'), 'w', encoding='utf-8') as _:
        json.dump(header_dict, _, ensure_ascii=False, indent=4)
    

    print('', file=sys.stdout)

    # ********************************************************************************
    # Z-BRAIN registration
    #
    # The Z-BRAIN suggestes the following registration script:
    #
    #    
    #    Source: https://zebrafishatlas.zib.de/tutorial/image-registration.html
    #    -----------------------------------------------------------------------------
    #
    #    [...]
    #    
    #    if [[ $mysetting == "testing" ]] ; then
    #      its=200x200x200
    #      percentage=0.2
    #      syn="20x20x0,1.e-8,5"
    #    else
    #      its=100x100x100
    #      percentage=0.25
    #      syn="100x100x50,1.e-8,5"
    #      mysetting=forproduction
    #    fi
    #    
    #    antsRegistration -d $dim -r [ $f, $m ,1] \
    #        -m mattes[  $f, $m , 1 , 32, regular, $percentage ] \
    #         -t translation[ 0.1 ] \
    #         -c [$its,1.e-8,20]  \
    #        -s 4x2x1vox  \
    #        -f 6x4x2 \
    #        -l 1 \
    #        -m mattes[  $f, $m , 1 , 32, regular, $percentage ] \
    #         -t rigid[ 0.1 ] \
    #         -c [$its,1.e-8,20]  \
    #        -s 4x2x1vox  \
    #        -f 3x2x1 \
    #        -l 1 \
    #        -m mattes[  $f, $m , 1 , 32, regular, $percentage ] \
    #         -t affine[ 0.1 ] \
    #         -c [$its,1.e-8,20]  \
    #        -s 4x2x1vox  \
    #        -f 3x2x1 \
    #        -l 1 \
    #        -m mattes[  $f, $m , 0.5 , 32 ] \
    #        -m cc[  $f, $m , 0.5 , 4 ] \
    #         -t SyN[ .1, 3, 0 ] \
    #         -c [ $syn ]  \
    #        -s 4x2x1vox  \
    #        -f 4x2x1 \
    #        -l 1 \
    #        -u 1 \
    #        -z 1 \
    #        -o [${nm},${nm}_diff.nii.gz,${nm}_inv.nii.gz]
    #
    #    -----------------------------------------------------------------------------
    #
    #
    # This script is implemented in the following:
    #
    # def ZBRAIN_registration(fixed, moving, outprefix=''):
    #     ITS = '100x100x100'
    #     PERCENTAGE = '0.25'
    #     SYN = '100x100x50,1.e-8,5'
    #     
    #     if False:
    #         ITS = '200x200x200'
    #         PERCENTAGE = '0.2'
    #         SYN = '20x20x0,1.e-8,5'	
    #     
    #     
    #     ZBRAIN_registration_args = [
    #         '--dimensionality',                  '{}'.format(len(fixed.shape)),
    #         '--initial-moving-transform',        '[{},{},1]'.format('<$f>', '<$m>'),
    #         
    #         # --- STAGE 1
    #         '--metric',                          'Mattes[{},{},1,32,Regular,{}]'.format('<$f>', '<$m>', PERCENTAGE),
    #         '--transform',                       'Translation[ 0.1 ]',
    #         '--convergence',                     '[ {},1.e-8,20 ]'.format(ITS),
    #         '--smoothing-sigmas',                '4x2x1vox',
    #         '--shrink-factors',                  '6x4x2',
    #         '--use-estimate-learning-rate-once', '1',
    #         # --- STAGE 2
    #         '--metric',                          'Mattes[{},{},1,32,Regular,{}]'.format('<$f>', '<$m>', PERCENTAGE),
    #         '--transform',                       'Rigid[ 0.1 ]',
    #         '--convergence',                     '[ {},1.e-8,20 ]'.format(ITS),
    #         '--smoothing-sigmas',                '4x2x1vox',
    #         '--shrink-factors',                  '3x2x1',
    #         '--use-estimate-learning-rate-once', '1',
    #         # --- STAGE 3
    #         '--metric',                          'Mattes[{},{},1,32,Regular,{}]'.format('<$f>', '<$m>', PERCENTAGE),
    #         '--transform',                       'Affine[ 0.1 ]',
    #         '--convergence',                     '[ {},1.e-8,20 ]'.format(ITS),
    #         '--smoothing-sigmas',                '4x2x1vox',
    #         '--shrink-factors',                  '3x2x1',
    #         '--use-estimate-learning-rate-once', '1',
    #         # --- STAGE 4
    #         '--metric',                          'Mattes[{},{},0.5,32]'.format('<$f>', '<$m>'),
    #         '--metric',                          'CC[{},{},0.5,4]'.format('<$f>', '<$m>'),
    #         '--transform',                       'SyN[ 0.1,3,0 ]',
    #         '--convergence',                     '[ {} ]'.format(SYN),
    #         '--smoothing-sigmas',                '4x2x1vox',
    #         '--shrink-factors',                  '4x2x1',
    #         '--use-estimate-learning-rate-once', '1',
    #         
    #         '--use-histogram-matching',          '1',
    #         '--collapse-output-transforms',      '1',
    #         '--write-composite-transform',       '1',
    #         
    #         #'--winsorize-image-intensities',     '[ 0.005,0.995 ]',
    #         
    #         '--float',                           '0',
    #         '--interpolation',                   'Linear',
    #         '--output',                          '[{},{},{}]'.format('<$outprefix>', '<$wmo>', '<$wfo>'),
    #         '--verbose',                         '1'
    #     ]
    #     
    #     return antsX.custom_registration(fixed, moving, ZBRAIN_registration_args, outprefix=outprefix)
    #
    #
    # kwargs = {
    #     'fixed': __reference_atlas.labels[atlas_registration_label],
    #     'moving': __moving_image,
    #     'outprefix': os.path.join(moving_data_directory, 'registration', 'transformation_'),
    # }
    # 
    # ZBRAIN_registration(**kwargs)
    #
        

if __name__ == "__main__":
    # ********************************************************************************
    # Argument parsing
    #
    
    import argparse
    
    
    __parser = argparse.ArgumentParser(
        description='Registers a volume image to a reference atlas, such as one derived from a Z-BRAIN 1.0 dataset.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    __parser.add_argument('--reference-atlas-directory', dest='reference_atlas_directory', type=str, required=True, metavar='<path>', help='Path to the directory of a reference atlas.')
    __parser.add_argument('--atlas-registration-label', dest='atlas_registration_label', type=str, required=True, metavar='<name>', help='Name of an anatomical label defined in the reference atlas. The registration will be performed to this label.')
    __parser.add_argument('--moving-data-directory', dest='moving_data_directory', type=str, required=True, metavar='<path>', help='Path to the directory of a moving image.')
    __parser.add_argument('--moving-image', dest='moving_image', type=str, required=True, metavar='<path>', help='Moving image in the directory of the moving image. This image will be registered to the reference atlas.')
    __parser.add_argument('--run-initialisation', dest='initialisation', action='store_true', help='-- NOT IMPLEMENTED -- Indicates whether to perform a search for an initial transform to align the moving image with the reference atlas beform the actual registration. Note that performing the initialisation might take a considerable amount of time, but should improve the registration results. However, if the moving image is already properly oriented in space, this is generally not necessary.')
    
    kwargs = vars(__parser.parse_args())
    

    # ********************************************************************************
    # Preprocess arguments

    if kwargs['initialisation']:
        kwargs.pop('initialisation')

    # ********************************************************************************
    # Execute main function

    try:
        registration(**kwargs, verbose=True)
    except:
        print('', file=sys.stdout)
        print('', file=sys.stdout)
        print('An error occured. Operation could not be completed.', file=sys.stdout)
        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1)

    sys.exit(0)
