
import class_cameracalib as ccc

# Call for multi images
cc = ccc.cameracalib(mode='multi', deblev=2)
cc.run(input="camera_cal")            
