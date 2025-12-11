from PIL import Image

rtd_note_title = '#6ab0de'
rtd_note_body  = '#e7f2fa'

rtd_warning_title = '#f0b37e'
rtd_warning_body  = '#ffedcc'

def PIL_hstack(images, margin=5):
    widths, heights = zip(*(i.size for i in images))

    n_img = len(images)
    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGBA', (total_width + margin*(n_img-1), max_height))
    x_offset = 0

    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0] + margin

    return new_im
