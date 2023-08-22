import os
import subprocess
import filecmp
import logging
import shutil
import sysconfig

def sync_bits_and_bytes_files():
    import filecmp

    """
    Check for "different" bitsandbytes Files and copy only if necessary.
    This function is specific for Windows OS.
    """

    # Only execute on Windows
    if os.name != 'nt':
        print('This function is only applicable to Windows OS.')
        return

    try:
        print(f'Copying bitsandbytes files...')
        # Define source and destination directories
        source_dir = os.path.join(os.getcwd(), 'bitsandbytes_windows')

        dest_dir_base = os.path.join(
            sysconfig.get_paths()['purelib'], 'bitsandbytes'
        )

        # Clear file comparison cache
        filecmp.clear_cache()

        # Iterate over each file in source directory
        for file in os.listdir(source_dir):
            source_file_path = os.path.join(source_dir, file)

            # Decide the destination directory based on file name
            if file in ('main.py', 'paths.py'):
                dest_dir = os.path.join(dest_dir_base, 'cuda_setup')
            else:
                dest_dir = dest_dir_base

            dest_file_path = os.path.join(dest_dir, file)

            # Compare the source file with the destination file
            if os.path.exists(dest_file_path) and filecmp.cmp(
                source_file_path, dest_file_path
            ):
                print(
                    f'Skipping {source_file_path} as it already exists in {dest_dir}'
                )
            else:
                # Copy file from source to destination, maintaining original file's metadata
                print(f'Copy {source_file_path} to {dest_dir}')
                shutil.copy2(source_file_path, dest_dir)

    except FileNotFoundError as fnf_error:
        print(f'File not found error: {fnf_error}')
    except PermissionError as perm_error:
        print(f'Permission error: {perm_error}')
    except Exception as e:
        print(f'An unexpected error occurred: {e}')

sync_bits_and_bytes_files()
