import class_lane_lines as cll

ll = cll.lane_lines(mode='video', output='deluxe')
mov = ll.run(media='./Media/project_video.mp4')
mov.write_videofile(filename='./project_video_masked.mp4', audio=False)
