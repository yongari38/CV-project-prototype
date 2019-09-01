# -*- coding: utf-8 -*-
import face_recognition
import cv2
import os
from os.path import isfile, join
import urllib
import argparse
import requests
from bs4 import BeautifulSoup
import youtube_dl
requests.packages.urllib3.disable_warnings()

from atpbar import register_reporter, find_reporter, flush, atpbar
import time
import multiprocessing as mp


def make_DIR(PATH):
    try:
        if not os.path.exists(PATH):
            os.makedirs(PATH)
    except OSError:
            print ('Error: Creating directory of '+PATH)
    
def getYoutubeURL(keyword, number):
    page = 1 # target search query page
    search_url = 'https://www.youtube.com/results?sp=EgIQAQ%253D%253D&search_query={}&page={}'
    search_soup = BeautifulSoup(requests.get(search_url.format(keyword, page), verify = False).text, 'html.parser')
    search_results = search_soup.find_all('div', {'class' : 'yt-lockup-content'})
    video_url_list = []
    for search_result in search_results[:number]:
        video_url = 'https://www.youtube.com' + search_result.h3.a['href']
        print(video_url + ' - '+ search_result.h3.a.text)
        video_url_list.append(video_url)
    return video_url_list


def youtubeDownload(urlList, vid_DIR):
    try:
        os.stat(vid_DIR)
    except:
        os.mkdir(vid_DIR)
    os.chdir(vid_DIR)

    # modify opts to customize output format
    opts = {
        # "http://www.youtube.com/watch?v=%(id)s.%(ext)s" but illegal filename format '/'
        'outtmpl': '%(id)s.%(ext)s'
        }
    youtube_dl.YoutubeDL(opts).download(urlList)

    os.chdir("..")
    return 0
    
    
def inputFaceEncode(imagePATH):
    # Load sample images of face and encode, can simply add more
    face_image = face_recognition.load_image_file(imagePATH)
    face_encoded = face_recognition.face_encodings(face_image)[0]
    
    known_faces.append(face_encoded)
    return len(known_faces)
    

def faceRec(f, known_faces):
    make_DIR("hit_frame/"+f)
    vid_PATH = "vid_DIR/"+f
    input_movie = cv2.VideoCapture(vid_PATH)
    frame_length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_length = int(input_movie.get(cv2.CAP_PROP_FPS))

    face_locations = []
    face_encodings = []
    face_names = []
    frame_number = 0
    
    for i in atpbar(range(frame_length), f):
        hit = False
        next, frame = input_movie.read()
        frame_number += 1
        
        if(int(input_movie.get(1)%(fps_length/2))==0): # 2 trials per sec

            # Convert the image from BGR color (OpenCV) to RGB color (face_recognition)
            rgb_frame = frame[:, :, ::-1]

            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.50)

                name = None
				# ToDo: automaticly parse name from known_faces
                if match[0]:
                    name = "Allen"
                    hit = True
                elif match[1]:
                    name = "Hoyong"
                    hit = True
                    
                face_names.append(name)
        
            # Label the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                if not name:
                    continue
        
                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 1)

				# Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
                
        
            #print("Processing frame {} / {}".format(frame_number, frame_length))
            if hit:
                cv2.imwrite('hit_frame/{}/{}.jpg'.format(f,frame_number),frame)
                #print("hit!")
    #################################
    '''
    next = True
    while (next):
        hit = False
        # Grab a single frame of video
        next, frame = input_movie.read()
        frame_number += 1
        
        if(int(input_movie.get(1)%(fps_length/2))==0): # 2 trials per sec

            # Convert the image from BGR color (OpenCV) to RGB color (face_recognition)
            rgb_frame = frame[:, :, ::-1]

            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.50)

                name = None
				# ToDo: automaticly parse name from known_faces
                if match[0]:
                    name = "Allen"
                    hit = True
                elif match[1]:
                    name = "Hoyong"
                    hit = True
                    
                face_names.append(name)
        
            # Label the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                if not name:
                    continue
        
                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 1)

				# Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
                
        
            #print("Processing frame {} / {}".format(frame_number, frame_length))
            if hit:
                cv2.imwrite('hit_frame/{}/{}.jpg'.format(f,frame_number),frame)
                #print("hit!")
		'''
    # All done!
    #print('{} done'.format(f))
    input_movie.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
	start = time.time()
	keyword = "allen"
	cnt = 5
	urlList = getYoutubeURL(keyword,cnt)
	make_DIR("vid_DIR")
	youtubeDownload(urlList,"vid_DIR")

	print("encoding face...")
	known_faces = []
	inputFaceEncode("sample/allen.jpg")
	inputFaceEncode("sample/hoyong2.jpg")
	print("encoding face complete")
	# ...

	onlyfiles = [f for f in os.listdir("vid_DIR") if isfile(join("vid_DIR", f))]
	

	'''
	# single pipeline
	make_DIR("hit_frame")
	for f in onlyfiles:
		faceRec(f,known_faces)
		'''
	
	# multiprocessing
	mp.set_start_method('spawn')
	
	procs = []
	
	for f in onlyfiles:
		proc = mp.Process(target=faceRec, args=(f, known_faces))
		proc.start()
	for proc in procs:
		proc.join()
	
	'''
	###############################################################
	def sampletask(n, name):
		for i in atpbar(range(10000), name=name):
			time.sleep(0.0001)
			
	def worker(reporter, task, queue):
		register_reporter(reporter)
		while True:
			args = queue.get()
			if args is None:
				queue.task_done()
				break
			task(*args)
			queue.task_done()
			
	nprocesses = mp.cpu_count()
	ntasks = cnt
	reporter = find_reporter()
	queue = mp.JoinableQueue()
	for i in range(nprocesses):
		p = mp.Process(target=worker, args=(reporter, faceRec, queue))
		p.start()
	for f in onlyfiles:
		queue.put((f,known_faces))
	for i in range(nprocesses):
		queue.put(None)
		queue.join()
	flush()
    #############################################
	'''
	end = time.time()
	print(end-start)
	print(end-start)
	print(end-start)
	print(end-start)
	print(end-start)
	print(end-start)
	print(end-start)
	print(end-start)
