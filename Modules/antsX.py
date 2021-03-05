# ******************************************************************************
# This file contains functions extending the functionality of the ANTsPy medical
# imaging library.
#
# The code shall be licensed under the GNU General Public License v3.0.
#
# (c) 2020- , Jan Mölter
# ******************************************************************************

__all__ = ['multistage_registration',
           'custom_registration',
           'SyNQuick_registration',
           'header_info',
          ]


import ants

import numpy as np

import re
import glob
import tempfile


def header_info(image):
    """
    Returns the header information of an ANTs image.

    Parameters
    ----------
    image : ants.core.ants_image.ANTsImage
        Fixed image.

    Returns
    -------
    _ : dict
        Header information.
    """
    if type(image) is not ants.core.ants_image.ANTsImage:
        raise TypeError('`image` is expected to be of type ants.core.ants_image.ANTsImage.')
        
    __header_info = dict()
    
    __header_info['dimensions'] = image.shape
    __header_info['spacing'] = image.spacing
    __header_info['origin'] = image.origin
    __header_info['orientation'] = image.orientation
    __header_info['direction'] = image.direction
    
    return __header_info

def direction_matrix(orientation):
    """
    Returns the direction matrix corresponding to orientation specification.
    
    Parameters
    ----------
    orientation : str
        Orientation specifier.

    Returns
    -------
    _ : (3,3) ndarray
        Direction matrix.
    """
    
    if type(orientation) is not str:
        raise TypeError('`orientation` is expected to be of type str.')
        
    if not orientation in ants.get_possible_orientations():
        raise ValueError('`orientation` is invalid.')
    
    __IMAGE = ants.from_numpy(np.zeros((1,1,1), dtype='uint8'))
    __IMAGE = __IMAGE.reorient_image2(orientation=orientation)
    
    return __IMAGE.direction


def multistage_registration(fixed, moving, transforms_kwargs, warpedout=False):
    """
    Performs a registration across multiple stages, where the results are properly
    propagated through the different stages.

    Parameters
    ----------
    fixed : ants.core.ants_image.ANTsImage
        Fixed image.
    moving : ants.core.ants_image.ANTsImage
        Moving image.
    transforms_kwargs : list of dict
        List of transformation parameters.
    warpedout: bool
        Include warped output image. Default is False.

    Returns
    -------
    _ : dict
        ants.registration.interface.registration like output.
    """

    #transforms_kwargs = [
    #    {
    #        'type_of_transform': 'Translation',
    #        'verbose': True
    #    }
    #    ,
    #    {
    #        'type_of_transform': 'Rigid',
    #        'verbose': True
    #    }
    #    ,
    #    {
    #        'type_of_transform': 'Affine',
    #        'verbose': True
    #    }
    #    ,
    #    {
    #        'type_of_transform': 'SyNOnly',
    #        'verbose': True
    #    }
    #]
    #

    if type(fixed) is not ants.core.ants_image.ANTsImage:
        raise TypeError('`fixed` is expected to be of type ants.core.ants_image.ANTsImage.')
    if type(moving) is not ants.core.ants_image.ANTsImage:
        raise TypeError('`moving` is expected to be of type ants.core.ants_image.ANTsImage.')
        

    __warpedmovout = moving
    
    __fwdtransforms = []
    __invtransforms = []
    
    for transform_kwargs in transforms_kwargs:
        __T = ants.registration(fixed=fixed, moving=__warpedmovout, **transform_kwargs)
        
        __warpedmovout = __T['warpedmovout']
        __fwdtransforms = __T['fwdtransforms'] + __fwdtransforms
        __invtransforms = __invtransforms + __T['invtransforms']
    
    __whichtoinvert = [('GenericAffine' in t) for t in __invtransforms]
    
    __warpedmovout = None
    __warpedfixout = None
    if warpedout:
        __warpedmovout = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=__fwdtransforms, interpolator='linear')
        __warpedfixout = ants.apply_transforms(fixed=moving, moving=fixed, transformlist=__invtransforms, whichtoinvert=__whichtoinvert, interpolator='linear')
        
    return {
        'warpedmovout': __warpedmovout,
        'warpedfixout': __warpedfixout,
        'fwdtransforms': __fwdtransforms,
        'invtransforms': __invtransforms,
        'whichtoinvert': __whichtoinvert
    }

def custom_registration(fixed, moving, args, outprefix=''):
    """
    Implements a direct interface to the full ANTs registration procedure in terms
    of the parameters, yet behaves similar to the ANTsPy functions and produces
    similar output.

    Parameters
    ----------
    fixed : ants.core.ants_image.ANTsImage
        Fixed image.
    moving : ants.core.ants_image.ANTsImage
        Moving image.
    args : list of str
        ANTs registration command line arguments

    Returns
    -------
    _ : dict
        ants.registration.interface.registration like output.
    """

    if type(fixed) is not ants.core.ants_image.ANTsImage:
        raise TypeError('`fixed` is expected to be of type ants.core.ants_image.ANTsImage.')
    if type(moving) is not ants.core.ants_image.ANTsImage:
        raise TypeError('`moving` is expected to be of type ants.core.ants_image.ANTsImage.')
    if not (type(args) is list and all([type(_) is str for _ in args])):
        raise TypeError('`args` is expected to be of type list with elements of type str.')
    

    moving = moving.clone('float')
    fixed = fixed.clone('float')
    warpedfixout = moving.clone()
    warpedmovout = fixed.clone()
    
    inpixeltype = fixed.pixeltype
    
    f = ants.utils.get_pointer_string(fixed)
    m = ants.utils.get_pointer_string(moving)
    wfo = ants.utils.get_pointer_string(warpedfixout)
    wmo = ants.utils.get_pointer_string(warpedmovout)
    

    if outprefix == '' or len(outprefix) == 0:
        outprefix = tempfile.mktemp()

    
    for i, arg in enumerate(args):
        if ('<$f>' in arg) or ('<$m>' in arg) or ('<$wmo>' in arg) or ('<$wfo>' in arg) or ('<$outprefix>' in arg):
            args[i] = arg.replace('<$f>', f).replace('<$m>', m).replace('<$wmo>', wmo).replace('<$wfo>', wfo).replace('<$outprefix>', outprefix)
    
    
    write_composite_transform = False
    for switch in ['--write-composite-transform', '-a']:
        if switch in args and args[args.index(switch)+1] == '1':
            write_composite_transform = True
            break
    
    
    #print(args)
    #return 0
    ants.registration(args, None)

    # ********************************************************************************
    # The code below has been taken from `ants.registration.interface.registration` to
    # to make this registration function behave and produce similar output as the
    # original function `ants.registration.interface.registration`.
    
    afffns = glob.glob(outprefix + "*" + "[0-9]GenericAffine.mat")
    fwarpfns = glob.glob(outprefix + "*" + "[0-9]Warp.nii.gz")
    iwarpfns = glob.glob(outprefix + "*" + "[0-9]InverseWarp.nii.gz")
    vfieldfns = glob.glob(outprefix + "*" + "[0-9]VelocityField.nii.gz")
    # print(afffns, fwarpfns, iwarpfns)
    if len(afffns) == 0:
        afffns = ""
    if len(fwarpfns) == 0:
        fwarpfns = ""
    if len(iwarpfns) == 0:
        iwarpfns = ""
    if len(vfieldfns) == 0:
        vfieldfns = ""

    alltx = sorted(
        set(glob.glob(outprefix + "*" + "[0-9]*"))
        - set(glob.glob(outprefix + "*VelocityField*"))
    )
    findinv = np.where(
        [re.search("[0-9]InverseWarp.nii.gz", ff) for ff in alltx]
    )[0]
    findfwd = np.where([re.search("[0-9]Warp.nii.gz", ff) for ff in alltx])[
        0
    ]
    if len(findinv) > 0:
        fwdtransforms = list(
            reversed(
                [ff for idx, ff in enumerate(alltx) if idx != findinv[0]]
            )
        )
        invtransforms = [
            ff for idx, ff in enumerate(alltx) if idx != findfwd[0]
        ]
    else:
        fwdtransforms = list(reversed(alltx))
        invtransforms = alltx

    if write_composite_transform:
        fwdtransforms = outprefix + "Composite.h5"
        invtransforms = outprefix + "InverseComposite.h5"

    if not vfieldfns:
        return {
            "warpedmovout": warpedmovout.clone(inpixeltype),
            "warpedfixout": warpedfixout.clone(inpixeltype),
            "fwdtransforms": fwdtransforms,
            "invtransforms": invtransforms,
        }
    else:
        return {
            "warpedmovout": warpedmovout.clone(inpixeltype),
            "warpedfixout": warpedfixout.clone(inpixeltype),
            "fwdtransforms": fwdtransforms,
            "invtransforms": invtransforms,
            "velocityfield": vfieldfns,
        }


def SyNQuick_registration(fixed, moving, outprefix=''):
    """
    Performs the SyNQuick[s] registration.

    Parameters
    ----------
    fixed : ants.core.ants_image.ANTsImage
        Fixed image.
    moving : ants.core.ants_image.ANTsImage
        Moving image.
    outprefix : str
        Output prefix.

    Returns
    -------
    _ : dict
        ants.registration.interface.registration like output.
    """

    if type(fixed) is not ants.core.ants_image.ANTsImage:
        raise TypeError('`fixed` is expected to be of type ants.core.ants_image.ANTsImage.')
    if type(moving) is not ants.core.ants_image.ANTsImage:
        raise TypeError('`moving` is expected to be of type ants.core.ants_image.ANTsImage.')
    
    
    RIGIDCONVERGENCE = '[ 1000x500x250x0,1e-6,10 ]'
    RIGIDSHRINKFACTORS = '8x4x2x1'
    RIGIDSMOOTHINGSIGMAS = '3x2x1x0vox'
    
    AFFINECONVERGENCE = '[ 1000x500x250x0,1e-6,10 ]'
    AFFINESHRINKFACTORS = '8x4x2x1'
    AFFINESMOOTHINGSIGMAS = '3x2x1x0vox'
    
    SYNCONVERGENCE = '[ 100x70x50x0,1e-6,10 ]'
    SYNSHRINKFACTORS = '8x4x2x1'
    SYNSMOOTHINGSIGMAS = '3x2x1x0vox'
    
    if any([_ > 256 for _ in list(fixed.shape)]):
        RIGIDCONVERGENCE = '[ 1000x500x250x0,1e-6,10 ]'
        RIGIDSHRINKFACTORS = '12x8x4x2'
        RIGIDSMOOTHINGSIGMAS = '4x3x2x1vox'
    
        AFFINECONVERGENCE = '[ 1000x500x250x0,1e-6,10 ]'
        AFFINESHRINKFACTORS = '12x8x4x2'
        AFFINESMOOTHINGSIGMAS = '4x3x2x1vox'
    
        SYNCONVERGENCE = '[ 100x100x70x50x0,1e-6,10 ]'
        SYNSHRINKFACTORS = '10x6x4x2x1'
        SYNSMOOTHINGSIGMAS = '5x3x2x1x0vox'
    
        
    SyNQuick_registration_args = [
        '--dimensionality',                  '{}'.format(len(fixed.shape)),
        '--initial-moving-transform',        '[{},{},1]'.format('<$f>', '<$m>'),
        
        # --- STAGE 1
        '--transform',                       'Rigid[ 0.1 ]',
        '--metric',                          'MI[{},{},1,32,Regular,0.25]'.format('<$f>', '<$m>'),
        '--convergence',                     RIGIDCONVERGENCE,
        '--shrink-factors',                  RIGIDSHRINKFACTORS,
        '--smoothing-sigmas',                RIGIDSMOOTHINGSIGMAS,
        # --- STAGE 2
        '--transform',                       'Affine[ 0.1 ]',
        '--metric',                          'MI[{},{},1,32,Regular,0.25]'.format('<$f>', '<$m>'),
        '--convergence',                     AFFINECONVERGENCE,
        '--shrink-factors',                  AFFINESHRINKFACTORS,
        '--smoothing-sigmas',                AFFINESMOOTHINGSIGMAS,
        # --- STAGE 3
        '--transform',                       'SyN[ 0.1,3,0 ]',
        '--metric',                          'MI[{},{},1,32]'.format('<$f>', '<$m>'),
        '--convergence',                     SYNCONVERGENCE,
        '--shrink-factors',                  SYNSHRINKFACTORS,
        '--smoothing-sigmas',                SYNSMOOTHINGSIGMAS,
        
        '--float',                           '0',
        #'--collapse-output-transforms',      '1',
        '--write-composite-transform',       '1',
        '--interpolation',                   'Linear',
        '--use-histogram-matching',          '0',
        '--winsorize-image-intensities',     '[ 0.005,0.995 ]',
        '--output',                          '[{},{},{}]'.format('<$outprefix>', '<$wmo>', '<$wfo>'),
        '--verbose',                         '1'
    ]
    
    
    return custom_registration(fixed, moving, SyNQuick_registration_args, outprefix=outprefix)
