# -*- coding: utf-8 -*-
#!/usr/bin/python3

# Detect objects on a picture using
# Intel® Movidius™ Neural Compute Stick (NCS)

import os
import cv2
import sys
import numpy
import ntpath

import mvnc.mvncapi as mvnc
from time import localtime, strftime

import visualize_output
import deserialize_output

class NCS:

    device = None
    graph = None
    CONFIDANCE_TRESHOLD = 0.60
    INTEREST_CLASS = 15 # person
    graph_path = 'MobileNetSSD/graph'
    dims = [300, 300]
    mean = [127.5, 127.5, 127.5]
    scale = 0.00789
    colormode = "bgr"

    def __init__(self):
        self.device = self.open_ncs_device()
        self.graph = self.load_graph()

    def open_ncs_device(self):

    # Look for enumerated NCS device(s); quit program if none found.
        devices = mvnc.EnumerateDevices()
        if len( devices ) == 0:
            print( "No devices found" )
            quit()

        # Get a handle to the first enumerated device and open it
        device = mvnc.Device( devices[0] )
        try:
            device.OpenDevice()
        except:
            print("Error - Could not open NCS device")
            quit()

        return device

# ---- Step 2: Load a graph file onto the NCS device -------------------------

    def load_graph( self, device ):

        # Read the graph file into a buffer
        with open( self.graph_path, mode='rb' ) as f:
            blob = f.read()

        # Load the graph buffer into the NCS
        graph = self.device.AllocateGraph( blob )

        return graph

    # ---- Step 3: Pre-process the image ----------------------------------------

    def pre_process_image( self, frame ):

        # Resize image [Image size is defined by choosen network, during training]
        img = cv2.resize( frame, tuple( self.dims ) )

        # Convert RGB to BGR [OpenCV reads image in BGR, some networks may need RGB]
        if( self.colormode == "rgb" ):
            img = img[:, :, ::-1]

        # Mean subtraction & scaling [A common technique used to center the data]
        img = img.astype( numpy.float16 )
        img = ( img - numpy.float16( self.mean ) ) * self.scale

        return img

    # ---- Step 4: Read & print inference results from the NCS -------------------

    def infer_image( self, frame ):

        img = self.pre_process_image(frame)

        # Load the image as a half-precision floating point array
        self.graph.LoadTensor( img, 'user object' )

        # Get the results from NCS
        output, userobj = self.graph.GetResult()

        # Get execution time
        inference_time = self.graph.GetGraphOption( mvnc.GraphOption.TIME_TAKEN )

        # Deserialize the output into a python dictionary
        output_dict = deserialize_output.ssd( 
                        output, 
                        self.CONFIDANCE_THRESHOLD, 
                        frame.shape )

        # Print the results (each image/frame may have multiple objects)
        for i in range( 0, output_dict['num_detections'] ):

            # Filter a specific class/category
            if( output_dict.get( 'detection_classes_' + str(i) ) == self.INTEREST_CLASS ):

                cur_time = strftime( "%Y_%m_%d_%H_%M_%S", localtime() )
                print( "Person detected on " + cur_time )

                # Extract top-left & bottom-right coordinates of detected objects 
                (y1, x1) = output_dict.get('detection_boxes_' + str(i))[0]
                (y2, x2) = output_dict.get('detection_boxes_' + str(i))[1]

                # Prep string to overlay on the image
                display_str = ( 
                    self.labels[output_dict.get('detection_classes_' + str(i))]
                    + ": "
                    + str( output_dict.get('detection_scores_' + str(i) ) )
                    + "%" )

                # Overlay bounding boxes, detection class and scores
                frame = visualize_output.draw_bounding_box( 
                            y1, x1, y2, x2, 
                            frame,
                            thickness=4,
                            color=(255, 255, 0),
                            display_str=display_str )

                # Capture snapshots
                photo = ( os.path.dirname(os.path.realpath(__file__))
                        + "/captures/photo_"
                        + cur_time + ".jpg" )
                cv2.imwrite( photo, frame )

        # If a display is available, show the image on which inference was performed
        if 'DISPLAY' in os.environ:
            cv2.imshow( 'NCS inference', frame )

    # ---- Step 5: Unload the graph and close the device -------------------------

    def close_ncs_device(self):
        self.graph.DeallocateGraph()

        try:
            self.device.CloseDevice()
        except:
            print("Error - could not close NCS device.")
            quit()

        cv2.destroyAllWindows()
        
    