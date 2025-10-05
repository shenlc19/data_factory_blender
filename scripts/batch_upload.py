import os, glob, natsort

render_folders = natsort.natsorted(
    glob.glob('datasets/tmp512/renders/*')
)[:100]

for folder in render_folders:
    os.system(f'~/tosutil cp -r {folder} tos://cwm-backup/public_data/rendering_data/carverse_60views_even')