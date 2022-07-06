from preprocess import preprocesses


input_datadir = './train_img'
output_datadir = './only_faceimages'

obj = preprocesses(input_datadir,output_datadir)
nrof_images_total, nrof_successfully_aligned = obj.collect_data()

print('Total number of images: %d' % nrof_images_total)
print('Number of successfully aligned images: %d' % nrof_successfully_aligned)




