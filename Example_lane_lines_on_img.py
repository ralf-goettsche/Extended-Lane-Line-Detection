import class_lane_lines as cll

ll = cll.lane_lines(mode='picture', output='deluxe', debuglevel=5)
pic = ll.run(media='./Media/vlcsnap-05.jpg')
ll.plotimg(pic)
