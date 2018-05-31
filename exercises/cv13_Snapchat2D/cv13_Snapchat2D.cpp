//#############################################################################
//  File:      cv13_Snapchat2D.cpp
//  Purpose:   Minimal OpenCV app for a 2D Snapchatfilter
//  Taken from Satya Mallic on: http://www.learnopencv.com
//  Date:      Authumn 2017
//#############################################################################

#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
 
using namespace std;
using namespace cv;
using namespace cv::face;


//-----------------------------------------------------------------------------
static void drawDelaunay(Mat& img, Subdiv2D& subdiv, Scalar delaunay_color)
{
    vector<Vec6f> triangleList;
    subdiv.getTriangleList(triangleList);
    vector<Point> pt(3);
    Size size = img.size();
    Rect rect(0,0, size.width, size.height);

    for(size_t i = 0; i < triangleList.size(); i++)
    {
        Vec6f t = triangleList[i];
        pt[0] = Point(cvRound(t[0]), cvRound(t[1]));
        pt[1] = Point(cvRound(t[2]), cvRound(t[3]));
        pt[2] = Point(cvRound(t[4]), cvRound(t[5]));

        // Draw rectangles completely inside the image.
        if (rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2]))
        {   line(img, pt[0], pt[1], delaunay_color, 1, CV_AA, 0);
            line(img, pt[1], pt[2], delaunay_color, 1, CV_AA, 0);
            line(img, pt[2], pt[0], delaunay_color, 1, CV_AA, 0);
        }
    }
}
//-----------------------------------------------------------------------------
static void createDelaunay(Mat& img,
                           Subdiv2D& subdiv,
                           vector<Point2f>& points,
                           bool drawAnimated,
                           vector<vector<int>>& triangleIndexes)
{
    // Insert points into subdiv
    for (Point2f p : points)
    {
        subdiv.insert(p);

        if (drawAnimated)
        {   Mat img_copy = img.clone();
            drawDelaunay(img_copy, subdiv, Scalar(255, 255, 255));
            imshow("Delaunay Triangulation", img_copy);
            waitKey(100);
        }
    }

    // Unfortunately we don't get the triangles by there original point indexes.
    // We only get them with their vertex coordinates.
    // So we have to map them again to get the triangles with their point indexes.
    Size size = img.size();
    Rect rect(0,0, size.width, size.height);
    vector<Vec6f> triangleList;
    subdiv.getTriangleList(triangleList);
    vector<Point2f> pt(3);
    vector<int> ind(3);

    for( size_t i = 0; i < triangleList.size(); i++ )
    {
        Vec6f t = triangleList[i];
        pt[0] = Point2f(t[0], t[1]);
        pt[1] = Point2f(t[2], t[3]);
        pt[2] = Point2f(t[4], t[5 ]);

        if (rect.contains(pt[0]) &&
            rect.contains(pt[1]) &&
            rect.contains(pt[2]))
        {
            for(int j = 0; j < 3; j++)
                for(size_t k = 0; k < points.size(); k++)
                    if(abs(pt[j].x - points[k].x) < 1.0 &&
                       abs(pt[j].y - points[k].y) < 1)
                        ind[j] = (int)k;

            triangleIndexes.push_back(ind);
        }
    }
}

//-----------------------------------------------------------------------------
int main()
{
    // Load Face Detector
    // Note for Visual Studio: You must set the Working Directory to $(TargetDir)
    // with: Right Click on Project > Properties > Debugging 
    CascadeClassifier faceDetector("../_data/opencv/haarcascades/haarcascade_frontalface_alt2.xml");
 
    // Create an instance of Facemark
    Ptr<Facemark> facemark = FacemarkLBF::create();
 
    // Load landmark detector
    facemark->loadModel("../_data/calibrations/lbfmodel.yaml");
 
    // Set up webcam for video capture
    VideoCapture cam(0);
     
    // Variable to store a video frame and its grayscale 
    Mat frame, gray;

    std::vector<cv::Scalar> colors = {
        cv::Scalar(47, 20, 249),
        cv::Scalar(39, 126, 250),
        cv::Scalar(73, 254, 241),
        cv::Scalar(64, 253, 103),
        cv::Scalar(252, 218, 94),
        cv::Scalar(249, 68, 67),
        cv::Scalar(144, 12, 155)
    };
     
    // Read a frame
    while(cam.read(frame))
    {
        // Convert frame to grayscale because faceDetector requires grayscale image
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // Detect faces
        vector<Rect> faces;
        int min = (int)(frame.rows*0.4f); // the bigger min the faster
        int max = (int)(frame.rows*0.8f); // the smaller max the faster
        cv::Size minSize(min, min);
        cv::Size maxSize(max, max);

        faceDetector.detectMultiScale(gray, faces, 1.1, 3, 0, minSize, maxSize);

        // Variable for landmarks.
        // Landmarks for one face is a vector of points
        // There can be more than one face in the image. Hence, we
        // use a vector of vector of points.
        vector<vector<Point2f>> landmarks;

        // Run landmark detector
        bool success = facemark->fit(frame, faces, landmarks);

        if(success && landmarks.size() >= 1) {
            try {
                // Draw rect of first face
                // rectangle(frame, faces[0], cv::Scalar(255, 0, 0), 2);

                // Keep bounding rectangle around face points
                Size size = frame.size();

                vector<Point2f> points;
                for(int j=0; j < 68; j++) {
                    points.push_back(landmarks[0][j]);
                }

                Rect rectFace = boundingRect(points);
                Point2f center(rectFace.x + rectFace.width/2,
                               rectFace.y + rectFace.height/2);

                // Add image border points
                points.push_back(Point2f(0, 0));
                points.push_back(Point2f(size.width / 2, 0));
                points.push_back(Point2f(size.width - 1, 0));
                points.push_back(Point2f(size.width - 1, size.height/2));
                points.push_back(Point2f(size.width - 1, size.height-1));
                points.push_back(Point2f(size.width / 2, size.height-1));
                points.push_back(Point2f(0, size.height-1));
                points.push_back(Point2f(0, size.height/2));

                // Create an instance of Subdiv2D
                Rect rect(0, 0, size.width, size.height);
                Subdiv2D subdiv(rect);

                // Create and draw the Delaunay triangulation
                vector<vector<int>> triIndexes1;

                createDelaunay(frame, subdiv, points, false, triIndexes1);
                // drawDelaunay(frame, subdiv, Scalar(255, 255, 255));

                float mouthHeight = cv::norm(points[51] - points[57]);
                float faceHeight = cv::norm(points[27] - points[33]);

                float ratio = mouthHeight / faceHeight;

                // check if mouth is open
                if (ratio > 0.5) {
                    float width = points[54].x - points[48].x;
                    float widthPerBow = width / colors.size();

                    int topY = points[48].y > points[54].y ? points[48].y : points[54].y;
                    int bottomY = size.height;
                    int x = points[48].x;

                    for (int i = 0; i < colors.size(); i++) {
                        cv::rectangle(frame, Point(x + i * widthPerBow, topY), Point(x + (i+1) * widthPerBow, bottomY), colors[i], cv::FILLED);
                    }
                }
                imshow("Snapchat", frame);

            } catch (cv::Exception ex) {
                cout << ex.what() << endl;
                imshow("Snapchat", frame);
            }
        }
        
        // Wait for key to exit loop
        if (waitKey(10) != -1)
            return 0;
    }
    return 0;
}


//-----------------------------------------------------------------------------
