import class_cameracalib as ccc

# Call for single image
cc = ccc.cameracalib(mode='single', deblev=2)
cc.run(input="camera_cal\calibration3.jpg", outfile="calibration3_average.points.jpg")
