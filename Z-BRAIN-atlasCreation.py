import os
import sys

import traceback

from Modules import ZBRAIN


def ZBRAIN_atlas_generation(name, output_directory, Z_BRAIN_directory, spacing=None, verbose=None):
    
    if not os.path.isdir(Z_BRAIN_directory):
        raise FileNotFoundError('Z-BRAIN data directory does not exist.')
    if not os.path.isdir(output_directory):
        raise FileNotFoundError('Output directory does not exist.')
    
    ANATOMY_LABELS = {
        'Elavl3-GCaMP5G': 'Elavl3-GCaMP5G_6dpf_MeanImageOf7Fish',
        'Elavl3-H2BRFP': 'Elavl3-H2BRFP_6dpf_MeanImageOf10Fish'
    }
    
    MASKS = {
        'Telencephalon' : 'Telencephalon -',
        #'Telencephalon :: Anterior Commisure' : 'Telencephalon - Anterior Commisure',
        #'Telencephalon :: Optic Commissure' : 'Telencephalon - Optic Commissure',
        #'Telencephalon :: Postoptic Commissure' : 'Telencephalon - Postoptic Commissure',
        'Telencephalon :: Olfactory Bulb' : 'Telencephalon - Olfactory Bulb',
        'Telencephalon :: Pallium' : 'Telencephalon - Pallium',
        #'Telencephalon :: Subpallium' : 'Telencephalon - Subpallium',
        'Diencephalon' : 'Diencephalon -',
        'Diencephalon :: Habenula' : 'Diencephalon - Habenula',
        'Diencephalon :: Pineal' : 'Diencephalon - Pineal',
        #'Diencephalon :: Pituitary' : 'Diencephalon - Pituitary',
        #'Diencephalon :: Posterior Tuberculum' : 'Diencephalon - Posterior Tuberculum',
        #'Diencephalon :: Preoptic Area' : 'Diencephalon - Preoptic Area',
        #'Diencephalon :: Postoptic Commissure' : 'Diencephalon - Postoptic Commissure',
        #'Diencephalon :: Optic Chiasm' : 'Diencephalon - Optic Chiasm',
        'Diencephalon :: Pretectum' : 'Diencephalon - Pretectum',
        #'Diencephalon :: Dorsal Thalamus' : 'Diencephalon - Dorsal Thalamus',
        #'Diencephalon :: Ventral Thalamus' : 'Diencephalon - Ventral Thalamus',
        #'Diencephalon :: Eminentia Thalami' : 'Diencephalon - Eminentia Thalami',
        #'Diencephalon :: Caudal Hypothalamus' : 'Diencephalon - Caudal Hypothalamus',
        #'Diencephalon :: Intermediate Hypothalamus' : 'Diencephalon - Intermediate Hypothalamus',
        #'Diencephalon :: Rostral Hypothalamus' : 'Diencephalon - Rostral Hypothalamus',
        #'Diencephalon :: Torus Lateralis' : 'Diencephalon - Torus Lateralis',
        'Mesencephalon' : 'Mesencephalon -',
        #'Mesencephalon :: Medial Tectal Band' : 'Mesencephalon - Medial Tectal Band',
        'Mesencephalon :: Tectum Neuropil' : 'Mesencephalon - Tecum Neuropil',
        'Mesencephalon :: Tectum Stratum Periventriculare' : 'Mesencephalon - Tectum Stratum Periventriculare',
        'Mesencephalon :: Tegmentum' : 'Mesencephalon - Tegmentum',
        #'Mesencephalon :: Torus Longitudinalis' : 'Mesencephalon - Torus Longitudinalis',
        #'Mesencephalon :: Torus Semicircularis' : 'Mesencephalon - Torus Semicircularis',
        'Rhombencephalon' : 'Rhombencephalon -',
        #'Rhombencephalon :: Area Postrema' : 'Rhombencephalon - Area Postrema',
        'Rhombencephalon :: Cerebellum' : 'Rhombencephalon - Cerebellum',
        #'Rhombencephalon :: Eminentia Granularis' : 'Rhombencephalon - Eminentia Granularis',
        #'Rhombencephalon :: Inferior Olive' : 'Rhombencephalon - Inferior Olive',
        #'Rhombencephalon :: Corpus Cerebelli' : 'Rhombencephalon - Corpus Cerebelli',
        #'Rhombencephalon :: Valvula Cerebelli' : 'Rhombencephalon - Valvula Cerebelli',
        #'Ganglia :: Eyes' : 'Ganglia - Eyes',
        #'Ganglia :: Olfactory Epithelium' : 'Ganglia - Olfactory Epithelium'
    }
    
    FILES = {
        'reference': os.path.join(Z_BRAIN_directory, 'Ref20131120pt14pl2.nrrd'),
        'anatomy labels database': os.path.join(Z_BRAIN_directory, 'AnatomyLabelDatabase.hdf5'),
        'masks database': os.path.join(Z_BRAIN_directory, 'MaskDatabase.mat'),
        'extensions' : {
            'labels' : {},
            'masks': {
                'Hemispheres :: Right' : os.path.join(Z_BRAIN_directory, 'extensions', 'Hemispheres - Right.nrrd')
            },
        }
    }

    if verbose is not None and verbose:
        print('=== {} ==='.format('Create Z-BRAIN reference atlas'), file=sys.stdout)
        print(' Directory: {}'.format(os.path.join(output_directory, name)), file=sys.stdout)
    
    ZBRAIN.create_atlas(files=FILES, anatomy_labels=ANATOMY_LABELS, masks=MASKS, directory=os.path.join(output_directory, name), spacing=spacing)


if __name__ == "__main__":
    # ********************************************************************************
    # Usage:
    # > python _Z-BRAIN-atlas-creation.py --name <...>
    #                                     --output_directory <...>
    #                                     --Z_BRAIN_directory <...>
    #                                     --spacing <...>
    #
    
    import argparse
    
    
    __parser = argparse.ArgumentParser()
    __parser.add_argument('--name', type=str, required=True)
    __parser.add_argument('--output-directory', dest='output_directory', type=str, required=True)
    __parser.add_argument('--Z-BRAIN-directory', dest='Z_BRAIN_directory', type=str, required=True)
    #__parser.add_argument('--include-extensions', dest='include_specials', action='store_true', required=False)
    __parser.add_argument('--spacing', nargs='+', type=float, required=False)
    
    kwargs = vars(__parser.parse_args())

    # ********************************************************************************
    # Include default parameters

    if kwargs['spacing'] is not None:
        kwargs['spacing'] = tuple(kwargs['spacing'])

        if not len(kwargs['spacing']) == 3:
            kwargs['spacing'] = None

    # ********************************************************************************
    # Execute main function

    try:
        ZBRAIN_atlas_generation(**kwargs, verbose=True)
    except:
        print('An error occured. Operation could not be completed.', file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1)

    sys.exit(0)
