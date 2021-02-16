import os
import sys

import py7zr
import tifffile

import tempfile


def extract_7z_archive(archive_fname, output_directory):
    if not os.path.exists(archive_fname):
        raise FileNotFoundError('Archive does not exist.')
    else:
        if not os.path.splitext(archive_fname)[1].upper() in ['.7Z']:
            raise TypeError('The archive is expected to be in the 7z format.')
    
    
    TEMP_DIRECTORY = tempfile.mkdtemp()
    
    with py7zr.SevenZipFile(archive_fname, 'r') as archive:
        for file in archive.getnames():
            
            archive.extract(targets=file, path=TEMP_DIRECTORY)

            average_tiff_pages(tiff_fname=os.path.join(TEMP_DIRECTORY, file), output_directory=output_directory)

            os.remove(os.path.join(TEMP_DIRECTORY, file))
    
    print(TEMP_DIRECTORY)
    os.rmdir(TEMP_DIRECTORY)


def average_tiff_pages(tiff_fname, output_directory):
    if not os.path.exists(tiff_fname):
        raise FileNotFoundError('File does not exist.')
    else:
        if not os.path.splitext(tiff_fname)[1].upper() in ['.TIFF', '.TIF']:
            raise TypeError('The image file is expected to be in the TIFF format.')

    __IMAGE = None
    
    with tifffile.TiffFile(tiff_fname) as tiff:

        TIFF_PAGES = tiff.pages
        for n, page in enumerate(TIFF_PAGES):
            if n > 0:
                __IMAGE = n/(n+1) * __IMAGE + 1/(n+1) * page.asarray()
            else:
                __IMAGE = page.asarray()

    tifffile.imwrite(os.path.join(output_directory, r'{}__mean.tif'.format(os.path.splitext(os.path.basename(tiff_fname))[0])), (__IMAGE * 2**8).astype('uint16'), photometric='minisblack', compress=True)
    
    
if __name__ == "__main__":
    # ********************************************************************************
    # Usage:
    # > python summaryImageGeneration.py --7z-archive <...>
    #                                    --output-directory <...>
    #
    
    import argparse
    
    
    __parser = argparse.ArgumentParser()
    __parser.add_argument('--7z-archive', dest='archive_fname', default='', type=str, required=True)
    __parser.add_argument('--output-directory', dest='output_directory', default='', type=str, required=True)
    
    kwargs = vars(__parser.parse_args())

    # ********************************************************************************
    # Include default parameters


    # ********************************************************************************
    # Execute main function

    try:
        extract_7z_archive(**kwargs)
    except:
        print('An error occured. Operation could not be completed.', file=sys.stderr)
        sys.exit(1)

    sys.exit(0)
