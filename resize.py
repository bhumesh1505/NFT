from PIL import Image
# open an image file (.bmp,.jpg,.png,.gif) you have in the working folder
imageFile = "dm6.jpg"
im1 = Image.open(imageFile)
# adjust width and height to your needs
width = 600
height = 600
# use one of these filter options to resize the image
# im2 = im1.resize((width, height), Image.NEAREST)      # use nearest neighbour
# im3 = im1.resize((width, height), Image.BILINEAR)     # linear interpolation in a 2x2 environment
# im4 = im1.resize((width, height), Image.BICUBIC)      # cubic spline interpolation in a 4x4 environment
im5 = im1.resize((width, height), Image.ANTIALIAS)    # best down-sizing filter
ext = ".jpg"
# im2.save("NEAREST" + ext)
# im3.save("BILINEAR" + ext)
# im4.save("BICUBIC" + ext)
im5.save("tdm6" + ext)